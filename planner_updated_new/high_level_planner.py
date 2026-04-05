import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HighLevelPlanner:
    """
    Lightweight, Jetson-optimized task planner.
    Works with small LLMs like Phi-2 (llama.cpp).
    """

    SYSTEM_PROMPT = (
        "You are a robot task planner. Output ONLY a JSON array. No text before or after. "
        "Each item must be an object with keys: action, target, reason. "
        "Example: "
        '[{"action":"move","target":"bottle","reason":"go to it"},{"action":"pick","target":"bottle","reason":"pick up"}]. '
        "Valid actions: find, move, pick, place, scan, stop, avoid, reroute. "
        "Rules: if person detected use stop. If chair use avoid. "
        "For bring or fetch tasks, end with a move action back to start position. "
        "Output ONLY the JSON array. Nothing else. No explanation."
    )

    KEYWORD_RULES = [
        ("find", "find"), ("locate", "find"), ("where", "find"),
        ("move", "move"), ("go", "move"), ("navigate", "move"),
        ("pick", "pick"), ("grab", "pick"), ("take", "pick"),
        ("get", "pick"), ("bring", "pick"), ("fetch", "pick"),
        ("place", "place"), ("put", "place"), ("drop", "place"),
        ("deliver", "place"),
        ("scan", "scan"), ("look", "scan"), ("check", "scan"), ("survey", "scan"),
        ("wait", "wait"), ("stop", "wait"), ("hold", "wait"),
    ]

    def __init__(self, llm_client):
        self.llm_client = llm_client

    # ------------------------------------------------------------------ #
    #  PUBLIC
    # ------------------------------------------------------------------ #

    def plan(self, goal: str, context: Dict[str, Any]) -> List[Dict]:
        tasks = self._plan_with_llm(goal, context)

        if tasks:
            logger.info("HighLevelPlanner: LLM produced %d tasks", len(tasks))
            return tasks

        past = context.get("past_experience")
        if past:
            logger.warning("HighLevelPlanner: using past experience fallback")
            return past

        logger.warning("HighLevelPlanner: keyword fallback")
        return self._plan_with_keywords(goal)

    # ------------------------------------------------------------------ #
    #  LLM PATH
    # ------------------------------------------------------------------ #

    def _plan_with_llm(self, goal: str, context: Dict[str, Any]) -> Optional[List[Dict]]:
        prompt = self._build_prompt(goal, context)

        try:
            # ✅ Works with llama.cpp client
            if hasattr(self.llm_client, "invoke"):
                raw = self.llm_client.invoke(prompt)
            else:
                raw = self.llm_client.run(prompt)

        except Exception as e:
            logger.error("HighLevelPlanner: LLM call failed (%s)", e)
            return None

        if not raw:
            logger.warning("HighLevelPlanner: empty LLM output")
            return None

        return self._parse_json(raw)

    # ------------------------------------------------------------------ #
    #  PROMPT BUILDING
    # ------------------------------------------------------------------ #

    def _build_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        objects = context.get("objects", [])

        compact_objs = [
            {
                "label": o.get("label", "?"),
                # ✅ FIX: use position Z instead of missing "depth"
                "depth_m": round(o.get("position", [0, 0, 0])[2], 1),
            }
            for o in objects[:5]
        ]

        obj_str = json.dumps(compact_objs)

        count_note = f" ({len(objects)} total, showing 5)" if len(objects) > 5 else ""

        past = context.get("past_experience")
        if past and isinstance(past, list):
            past_str = "Past plan: " + json.dumps([t.get("action") for t in past])
        else:
            past_str = "No past experience."

        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Goal: {goal}\n\n"
            f"Visible objects{count_note}: {obj_str}\n\n"
            f"{past_str}\n\n"
            "JSON task array:"
        )

    # ------------------------------------------------------------------ #
    #  JSON PARSING (CRITICAL FIX)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(raw: str) -> Optional[List[Dict]]:
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

            # ✅ CRITICAL FIX: extract only JSON part
            start = cleaned.find("[")
            end = cleaned.rfind("]") + 1

            if start != -1 and end != -1:
                cleaned = cleaned[start:end]

            parsed = json.loads(cleaned)

            if isinstance(parsed, list) and parsed:
                return parsed

            if isinstance(parsed, dict):
                for key in ("tasks", "plan", "actions", "steps"):
                    if isinstance(parsed.get(key), list) and parsed[key]:
                        return parsed[key]

        except Exception as e:
            logger.warning(
                "HighLevelPlanner: JSON parse failed (%s)\nRAW:\n%s",
                e, raw[:300]
            )

        return None

    # ------------------------------------------------------------------ #
    #  KEYWORD FALLBACK
    # ------------------------------------------------------------------ #

    def _plan_with_keywords(self, goal: str) -> List[Dict]:
        goal_lower = goal.lower()
        tasks = []
        seen = set()

        for keyword, action in self.KEYWORD_RULES:
            if keyword in goal_lower and action not in seen:
                tasks.append({
                    "action": action,
                    "target": goal,
                    "reason": f"keyword: {keyword}",
                })
                seen.add(action)

        if not tasks:
            return [
                {"action": "scan", "target": "environment", "reason": "no match"},
                {"action": "ask_user", "target": "user", "reason": "unclear goal"},
            ]

        return tasks