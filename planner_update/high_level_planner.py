import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HighLevelPlanner:

    SYSTEM_PROMPT = (
        "You are the planning brain of a humanoid mobile robot manipulator. "
        "You receive a goal and the current world state. "
        "Break the goal into a JSON list of task objects. "
        "Each task object must have: "
        '{"action": "<verb>", "target": "<object or location>", "reason": "<why>"}. '
        "Valid actions: find, move, pick, place, scan, ask_user, wait. "
        "Return ONLY valid JSON — no explanation, no markdown, no backticks."
    )

    # Keyword fallback when LLM is unavailable
    KEYWORD_RULES = [
        ("find",   {"action": "find",     "target": "unknown", "reason": "keyword match"}),
        ("move",   {"action": "move",     "target": "unknown", "reason": "keyword match"}),
        ("pick",   {"action": "pick",     "target": "unknown", "reason": "keyword match"}),
        ("place",  {"action": "place",    "target": "unknown", "reason": "keyword match"}),
        ("bring",  {"action": "pick",     "target": "unknown", "reason": "keyword match"}),
        ("get",    {"action": "pick",     "target": "unknown", "reason": "keyword match"}),
        ("scan",   {"action": "scan",     "target": "environment", "reason": "keyword match"}),
        ("look",   {"action": "scan",     "target": "environment", "reason": "keyword match"}),
    ]

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient instance (has .invoke(prompt) → str | None)
        """
        self.llm_client = llm_client

    def plan(self, goal: str, context: Dict[str, Any]) -> List[Dict]:
        """
        Generate a structured task list for the given goal and world context.

        Returns a list of task dicts, guaranteed non-empty.
        Falls back to rule-based planning if LLM is unavailable or returns bad JSON.
        """
        # 1 — Try LLM
        tasks = self._plan_with_llm(goal, context)
        if tasks:
            logger.info("HighLevelPlanner: LLM produced %d tasks", len(tasks))
            return tasks

        # 2 — Fallback: use past experience if available
        past = context.get("past_experience")
        if past:
            logger.warning("HighLevelPlanner: using cached past plan (LLM unavailable)")
            return past

        # 3 — Fallback: keyword rules
        logger.warning("HighLevelPlanner: using keyword rule fallback")
        return self._plan_with_keywords(goal)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _plan_with_llm(self, goal: str, context: Dict[str, Any]) -> Optional[List[Dict]]:
        prompt = self._build_prompt(goal, context)
        raw = self.llm_client.invoke(prompt)

        if not raw:
            logger.warning("HighLevelPlanner: LLM returned None")
            return None

        return self._parse_json(raw)

    def _build_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        # Summarise context to keep tokens low
        obj_list = context.get("objects", [])
        obj_summary = json.dumps(obj_list[:20], indent=2)   # cap at 20 objects
        detections = context.get("live_detections", [])
        det_summary = json.dumps(detections[:10], indent=2)
        past_exp = context.get("past_experience")
        past_note = f"Past experience for similar goal: {json.dumps(past_exp)}" if past_exp else "No past experience."

        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Known objects in world model ({len(obj_list)} total, showing first 20):\n{obj_summary}\n\n"
            f"Live detections ({len(detections)} total, showing first 10):\n{det_summary}\n\n"
            f"{past_note}\n\n"
            "Return JSON list of tasks:"
        )

    @staticmethod
    def _parse_json(raw: str) -> Optional[List[Dict]]:
        """Parse LLM output into a list of task dicts. Tolerant of minor formatting issues."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
            # LLM sometimes wraps list in {"tasks": [...]}
            if isinstance(parsed, dict):
                for key in ("tasks", "plan", "actions"):
                    if isinstance(parsed.get(key), list):
                        return parsed[key]
        except json.JSONDecodeError as e:
            logger.warning("HighLevelPlanner: JSON parse failed (%s) — raw: %s", e, raw[:200])

        return None

    def _plan_with_keywords(self, goal: str) -> List[Dict]:
        goal_lower = goal.lower()
        tasks = []
        for keyword, task_template in self.KEYWORD_RULES:
            if keyword in goal_lower:
                task = dict(task_template)
                # Try to extract target from goal string (simple heuristic)
                task["target"] = goal
                tasks.append(task)

        if not tasks:
            # Absolute last resort
            tasks = [
                {"action": "scan",     "target": "environment", "reason": "no match found"},
                {"action": "ask_user", "target": "user",         "reason": "cannot determine goal"},
            ]

        return tasks
