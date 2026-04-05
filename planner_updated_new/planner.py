import logging
import time
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from planner.schemas            import PlannerState
from planner.context_builder    import ContextBuilder
from planner.high_level_planner import HighLevelPlanner
from planner.task_planner       import TaskPlanner
from planner.action_planner     import SkillLibrary
from planner.executor_interface import ExecutorInterface
from planner.experience_memory  import ExperienceMemory
from planner.plan_push          import PlanPush

logger = logging.getLogger(__name__)


# NEW: simple clarification logic (lightweight, no LLM needed)
def needs_clarification(goal: str) -> str:
    goal = goal.lower()

    if "coffee" in goal and ("decaf" not in goal and "caffeinated" not in goal):
        return "Do you want caffeinated or decaf coffee?"

    return ""


class Planner:
    MAX_RETRIES = 3

    def __init__(self, llm_client, world_model, perception=None):
        self.world_model       = world_model
        self.experience_memory = ExperienceMemory()
        self.context_builder   = ContextBuilder(
            world_model=world_model,
            perception=perception,
            experience_memory=self.experience_memory,
        )
        self.high_level   = HighLevelPlanner(llm_client)
        self.task_planner = TaskPlanner()
        self.skills       = SkillLibrary()
        self.executor     = ExecutorInterface()
        self.plan_push    = PlanPush()

        self.graph = self._build_graph()
        logger.info("Planner: initialised and graph compiled")

    # ------------------------------------------------------------------ #
    # GRAPH
    # ------------------------------------------------------------------ #

    def _build_graph(self):
        graph = StateGraph(PlannerState)

        graph.add_node("build_context", self.build_context)
        graph.add_node("plan_tasks",    self.plan_tasks)
        graph.add_node("plan_skills",   self.plan_skills)
        graph.add_node("execute",       self.execute)
        graph.add_node("check_goal",    self.check_goal)
        graph.add_node("ask_user",      self.ask_user)

        graph.set_entry_point("build_context")

        graph.add_edge("build_context", "plan_tasks")
        graph.add_edge("plan_tasks",    "plan_skills")
        graph.add_edge("plan_skills",   "execute")
        graph.add_edge("execute",       "check_goal")

        # 🔥 FIX: loop back after asking
        graph.add_edge("ask_user", "build_context")

        graph.add_conditional_edges(
            "check_goal",
            self.goal_condition,
            {
                "done":  END,
                "retry": "build_context",
                "scan":  "build_context",
                "ask":   "ask_user",
                "stop":  END,
            }
        )

        return graph.compile()

    # ------------------------------------------------------------------ #
    # NODES
    # ------------------------------------------------------------------ #

    def build_context(self, state: PlannerState) -> PlannerState:
        if hasattr(self.world_model, "person_in_scene") and self.world_model.person_in_scene:
            logger.warning("Planner: person detected — waiting 2s")
            time.sleep(2.0)

        try:
            state["context"] = self.context_builder.build(goal=state["goal"])
        except Exception as e:
            logger.error("Context build failed: %s", e)
            state["context"] = {"objects": [], "object_count": 0}

        return state

    def plan_tasks(self, state: PlannerState) -> PlannerState:
        logger.info("Planner: planning tasks")

        #  NEW: clarification check BEFORE planning
        question = needs_clarification(state["goal"])
        if question:
            state["_route"] = "ask"
            state["clarification_question"] = question
            return state

        try:
            tasks = self.high_level.plan(state["goal"], state["context"])
            state["tasks"] = tasks
        except Exception as e:
            logger.error("Task planning failed: %s", e)
            state["tasks"] = [{"action": "scan", "target": "environment"}]
            state["failure_reason"] = str(e)

        return state

    def plan_skills(self, state: PlannerState) -> PlannerState:
        try:
            state["skills"] = self.task_planner.plan(state["tasks"])
        except Exception as e:
            logger.error("Skill planning failed: %s", e)
            state["skills"] = [{"skill": "scan_environment"}]

        return state

    def execute(self, state: PlannerState) -> PlannerState:
        results = []

        for skill in state["skills"]:
            skill_name = skill.get("skill", "")
            target     = skill.get("target")

            # ── navigate_to: resolve coords → push to Nav2 ──────────
            if skill_name == "navigate_to":
                push_result = self.plan_push.push(
                    target=target,
                    world_model=self.world_model,
                )
                results.append({
                    "action":   "navigate_to",
                    "status":   push_result["status"],
                    "nav_goal": push_result.get("nav_goal"),
                    "position": push_result.get("position"),
                    "error":    push_result.get("error"),
                })
                logger.info(
                    "Planner.execute: navigate_to '%s' → %s  goal=%s",
                    target, push_result["status"], push_result.get("nav_goal"),
                )
                if push_result["status"] == "failed":
                    error = push_result.get("error", "navigation failed")
                    state["failure_reason"] = error
                    # If object not found in world model — say it clearly
                    if "Could not resolve coordinates" in str(error):
                        target_name = target or "object"
                        msg = f"{target_name} is not visible. Please place it in front of the camera."
                        logger.warning("Planner: %s", msg)
                        self.executor.execute(
                            ["speak_question"], target=msg
                        )
                continue

            # ── all other skills: normal executor path ───────────────
            if not self.skills.has_skill(skill_name):
                continue

            actions = self.skills.get_actions(skill_name)
            res = self.executor.execute(actions, target=target)
            results.extend(res)

        state["execution_result"] = results
        return state

    def check_goal(self, state: PlannerState) -> PlannerState:
        if state.get("failure_reason"):
            state["_route"] = "retry"
            return state

        state["_route"] = "done"
        state["done"] = True
        return state

    def ask_user(self, state: PlannerState) -> PlannerState:
        question = state.get("clarification_question")

        if not question:
            question = "Can you clarify?"

        logger.info("Asking user: %s", question)

        result = self.executor.execute(
            ["speak_question", "wait_for_response"],
            target=question
        )

        user_response = ""
        if result:
            user_response = result[-1].get("response", "")

        logger.info("User said: %s", user_response)

        #  update goal
        if user_response:
            state["goal"] = state["goal"] + " " + user_response

        # cleanup
        state["clarification_question"] = None

        # continue execution
        state["_route"] = "build_context"
        state["done"] = False

        return state

    def goal_condition(self, state: PlannerState) -> str:
        return state.get("_route", "done")

    # ------------------------------------------------------------------ #

    def run(self, goal: str) -> Dict[str, Any]:
        state: PlannerState = {
            "goal": goal,
            "context": {},
            "tasks": [],
            "skills": [],
            "execution_result": [],
            "failure_reason": None,
            "clarification_question": None,
            "retry_count": 0,
            "scan_count": 0,
            "candidates": None,
            "waiting_for_user": False,
            "done": False,
            "_route": "done",
        }

        return self.graph.invoke(state)