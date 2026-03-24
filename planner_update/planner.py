import logging
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from .schemas import PlannerState
from .context_builder import ContextBuilder
from .high_level_planner import HighLevelPlanner
from .task_planner import TaskPlanner
from .action_planner import SkillLibrary
from .executor_interface import ExecutorInterface
from .experience_memory import ExperienceMemory

logger = logging.getLogger(__name__)


class Planner:
    """
    Main LangGraph orchestrator for the robot brain.

    Graph flow:
        build_context → plan_tasks → plan_skills → execute → check_goal
                                                                  ↓
                                                   "done" → END
                                                   "retry" → build_context  (max 3 retries)
                                                   "scan"  → build_context  (object not found)
                                                   "ask"   → ask_user → check_goal
                                                   "stop"  → END with error

    Situations handled:
        1. Happy path          — everything works, task completes
        2. Wrong object        — post-grasp verification fails, retry loop
        3. Object not found    — context empty, triggers scan → retry
        4. Ambiguous objects   — multiple candidates, LLM asks user
        5. LLM API failure     — falls back to memory → keywords → safe stop
        6. Executor failure    — action fails, retries up to MAX_RETRIES
    """

    MAX_RETRIES = 3

    def __init__(self, llm_client, world_model, perception=None):
        self.experience_memory = ExperienceMemory()
        self.context_builder = ContextBuilder(
            world_model=world_model,
            perception=perception,
            experience_memory=self.experience_memory,
        )
        self.high_level = HighLevelPlanner(llm_client)
        self.task_planner = TaskPlanner()
        self.skills = SkillLibrary()
        self.executor = ExecutorInterface()

        self.graph = self._build_graph()
        logger.info("Planner: initialised and graph compiled")

    # ------------------------------------------------------------------ #
    #  Graph construction
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        graph = StateGraph(PlannerState)

        graph.add_node("build_context",  self.build_context)
        graph.add_node("plan_tasks",     self.plan_tasks)
        graph.add_node("plan_skills",    self.plan_skills)
        graph.add_node("execute",        self.execute)
        graph.add_node("check_goal",     self.check_goal)
        graph.add_node("ask_user",       self.ask_user)

        graph.set_entry_point("build_context")

        graph.add_edge("build_context", "plan_tasks")
        graph.add_edge("plan_tasks",    "plan_skills")
        graph.add_edge("plan_skills",   "execute")
        graph.add_edge("execute",       "check_goal")
        graph.add_edge("ask_user",      "check_goal")

        graph.add_conditional_edges(
            "check_goal",
            self.goal_condition,
            {
                "done":  END,      # success
                "retry": "build_context",   # wrong obj / exec fail → retry
                "scan":  "build_context",   # not found → scan wider
                "ask":   "ask_user",        # ambiguous → clarify
                "stop":  END,      # max retries hit or unrecoverable
            }
        )

        return graph.compile()

    # ------------------------------------------------------------------ #
    #  Graph nodes
    # ------------------------------------------------------------------ #

    def build_context(self, state: PlannerState) -> PlannerState:
        logger.info("Planner [build_context]: building world context for goal='%s'", state["goal"])
        try:
            state["context"] = self.context_builder.build(goal=state["goal"])
        except Exception as e:
            logger.error("Planner [build_context]: failed (%s)", e)
            state["context"] = {"objects": [], "positions": {}, "relations": [],
                                "live_detections": [], "past_experience": None,
                                "object_count": 0, "has_past_experience": False}
        return state

    def plan_tasks(self, state: PlannerState) -> PlannerState:
        logger.info("Planner [plan_tasks]: planning tasks")
        try:
            tasks = self.high_level.plan(state["goal"], state["context"])
            state["tasks"] = tasks
            state["llm_fallback"] = getattr(self.high_level, "_last_was_fallback", False)
            logger.info("Planner [plan_tasks]: %d tasks generated", len(tasks))
        except Exception as e:
            logger.error("Planner [plan_tasks]: failed (%s)", e)
            state["tasks"] = [{"action": "scan", "target": "environment", "reason": f"error:{e}"}]
            state["failure_reason"] = f"plan_tasks_error:{e}"
        return state

    def plan_skills(self, state: PlannerState) -> PlannerState:
        logger.info("Planner [plan_skills]: mapping tasks to skills")
        try:
            state["skills"] = self.task_planner.plan(state["tasks"])
            logger.info("Planner [plan_skills]: %d skills queued", len(state["skills"]))
        except Exception as e:
            logger.error("Planner [plan_skills]: failed (%s)", e)
            state["skills"] = [{"skill": "scan_environment", "target": "environment",
                                 "reason": f"error:{e}", "confidence_required": 0.0}]
            state["failure_reason"] = f"plan_skills_error:{e}"
        return state

    def execute(self, state: PlannerState) -> PlannerState:
        logger.info("Planner [execute]: executing %d skills", len(state["skills"]))
        all_results = []

        for skill_dict in state["skills"]:
            skill_name = skill_dict.get("skill", "")
            target = skill_dict.get("target")
            conf_required = skill_dict.get("confidence_required", 0.0)

            if not self.skills.has_skill(skill_name):
                logger.warning("Planner [execute]: unknown skill '%s' — skipping", skill_name)
                continue

            actions = self.skills.get_actions(skill_name)
            logger.debug("Planner [execute]: skill=%s → %d actions", skill_name, len(actions))

            results = self.executor.execute(actions, target=target,
                                            confidence_required=conf_required)
            all_results.extend(results)

            # Check for any failed action in this skill
            failed = [r for r in results if r["status"] != "success"]
            if failed:
                state["failure_reason"] = (
                    f"execution_failed: skill={skill_name} "
                    f"action={failed[0]['action']} error={failed[0].get('error','unknown')}"
                )
                logger.error("Planner [execute]: %s", state["failure_reason"])
                break

        state["execution_result"] = all_results
        return state

    def check_goal(self, state: PlannerState) -> PlannerState:
        """
        Evaluate whether the goal was achieved.
        Sets state["_route"] to one of: done / retry / scan / ask / stop
        """
        retry_count = state.get("retry_count", 0)
        failure = state.get("failure_reason")
        candidates = state.get("candidates")
        context = state.get("context", {})

        # ── Max retries exceeded ──────────────────────────────────────
        if retry_count >= self.MAX_RETRIES:
            logger.error("Planner [check_goal]: max retries (%d) reached — stopping", self.MAX_RETRIES)
            state["done"] = True
            state["error"] = "max_retries_exceeded"
            state["_route"] = "stop"
            return state

        # ── Ambiguous — multiple candidates ──────────────────────────
        if candidates and len(candidates) > 1:
            logger.info("Planner [check_goal]: ambiguity detected — %d candidates", len(candidates))
            state["_route"] = "ask"
            state["waiting_for_user"] = True
            return state

        # ── Object not found in context ───────────────────────────────
        if context.get("object_count", 0) == 0 and not failure:
            scan_count = state.get("scan_count", 0)
            if scan_count >= self.MAX_RETRIES:
                logger.error("Planner [check_goal]: object not found after %d scans", scan_count)
                state["done"] = True
                state["error"] = "object_not_found"
                state["_route"] = "stop"
            else:
                logger.info("Planner [check_goal]: empty context — triggering wider scan (scan %d)", scan_count + 1)
                state["scan_count"] = scan_count + 1
                state["failure_reason"] = None
                state["_route"] = "scan"
            return state

        # ── Execution failure — retry ─────────────────────────────────
        if failure:
            logger.warning("Planner [check_goal]: failure='%s' retry=%d", failure, retry_count)
            state["retry_count"] = retry_count + 1
            state["failure_reason"] = None
            state["_route"] = "retry"
            return state

        # ── Success ───────────────────────────────────────────────────
        logger.info("Planner [check_goal]: goal achieved — storing to experience memory")
        try:
            self.experience_memory.store(
                goal=state["goal"],
                plan=state["tasks"],
                outcome="success",
            )
        except Exception as e:
            logger.warning("Planner [check_goal]: experience_memory.store failed (%s)", e)

        state["done"] = True
        state["error"] = None
        state["_route"] = "done"
        return state

    def ask_user(self, state: PlannerState) -> PlannerState:
        """
        Handle ambiguous situations by asking the user for clarification.
        Generates a natural question using the LLM and sends it to the audio output.
        """
        candidates = state.get("candidates", [])
        goal = state["goal"]

        # Build clarification question
        question = self._generate_clarification_question(goal, candidates)
        logger.info("Planner [ask_user]: '%s'", question)

        # Send to robot audio via executor
        self.executor.execute(["speak_question", "wait_for_response"], target=question)

        # Store this disambiguation in experience memory for future
        try:
            self.experience_memory.store(
                goal=goal,
                plan=state.get("tasks", []),
                outcome="disambiguation_requested",
            )
        except Exception as e:
            logger.warning("Planner [ask_user]: experience_memory.store failed (%s)", e)

        # Reset ambiguity flags — user will re-issue a clearer command
        state["candidates"] = None
        state["waiting_for_user"] = False
        state["failure_reason"] = None
        state["_route"] = "done"   # end this cycle, wait for new command
        state["done"] = True
        return state

    # ------------------------------------------------------------------ #
    #  Conditional edge
    # ------------------------------------------------------------------ #

    def goal_condition(self, state: PlannerState) -> str:
        route = state.get("_route", "done")
        logger.info("Planner [goal_condition]: routing → %s", route)
        return route

    # ------------------------------------------------------------------ #
    #  Public run
    # ------------------------------------------------------------------ #

    def run(self, goal: str) -> Dict[str, Any]:
        """
        Main entry point. Pass the user's instruction string.
        Returns the final PlannerState dict.
        """
        logger.info("Planner.run: goal='%s'", goal)

        initial_state: PlannerState = {
            "goal":             goal,
            "context":          {},
            "tasks":            [],
            "skills":           [],
            "current_skill":    {},
            "execution_result": [],
            "failure_reason":   None,
            "retry_count":      0,
            "scan_count":       0,
            "grasp_confidence": None,
            "candidates":       None,
            "waiting_for_user": False,
            "llm_fallback":     False,
            "done":             False,
            "error":            None,
            "_route":           "done",
        }

        try:
            final_state = self.graph.invoke(initial_state)
            if final_state.get("error"):
                logger.warning("Planner.run: completed with error='%s'", final_state["error"])
            else:
                logger.info("Planner.run: completed successfully")
            return final_state
        except Exception as e:
            logger.error("Planner.run: unhandled exception (%s)", e)
            initial_state["done"] = True
            initial_state["error"] = f"unhandled_exception:{e}"
            return initial_state

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _generate_clarification_question(self, goal: str, candidates: list) -> str:
        if not candidates:
            return f"I couldn't find what you asked for. Could you describe it differently?"

        if len(candidates) == 2:
            a = candidates[0].get("description", "the first option")
            b = candidates[1].get("description", "the second option")
            return f"I found two possibilities. Did you mean {a}, or {b}?"

        count = len(candidates)
        return (
            f"I found {count} similar objects. "
            "Could you be more specific — for example, mention the colour, size, or location?"
        )
