import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class TaskPlanner:
    """
    Converts high-level task dicts (from HighLevelPlanner) into
    skill dicts that SkillLibrary can look up.

    Input task format:
        {"action": "pick", "target": "red Nescafé tin", "reason": "..."}

    Output skill format:
        {"skill": "pick_object", "target": "red Nescafé tin", "confidence_required": 0.85}
    """

    # Maps LLM action verb → skill name + required confidence
    ACTION_TO_SKILL: Dict[str, Dict] = {
        "find":     {"skill": "scan_environment",  "confidence_required": 0.0},
        "scan":     {"skill": "scan_environment",  "confidence_required": 0.0},
        "move":     {"skill": "navigate_to",       "confidence_required": 0.0},
        "navigate": {"skill": "navigate_to",       "confidence_required": 0.0},
        "pick":     {"skill": "pick_object",       "confidence_required": 0.85},
        "grab":     {"skill": "pick_object",       "confidence_required": 0.85},
        "get":      {"skill": "pick_object",       "confidence_required": 0.85},
        "bring":    {"skill": "pick_object",       "confidence_required": 0.85},
        "place":    {"skill": "place_object",      "confidence_required": 0.70},
        "put":      {"skill": "place_object",      "confidence_required": 0.70},
        "ask_user": {"skill": "ask_user",          "confidence_required": 0.0},
        "wait":       {"skill": "wait",              "confidence_required": 0.0},
        "stop":       {"skill": "stop_for_human",   "confidence_required": 0.0},
        "avoid":      {"skill": "reroute_right",    "confidence_required": 0.0},
        "reroute":    {"skill": "reroute_right",    "confidence_required": 0.0},
        "return":     {"skill": "return_to_start", "confidence_required": 0.0},
        "come_back":  {"skill": "return_to_start", "confidence_required": 0.0},
    }

    def plan(self, tasks: List[Dict]) -> List[Dict]:
        """
        Convert a list of task dicts into a list of skill dicts.
        Unrecognised actions are logged and skipped.
        """
        skills = []

        for task in tasks:
            if not isinstance(task, dict):
                logger.warning("TaskPlanner: skipping non-dict task: %s", task)
                continue

            action = task.get("action", "").lower().strip()
            target = task.get("target", "unknown")
            reason = task.get("reason", "")

            if action not in self.ACTION_TO_SKILL:
                logger.warning("TaskPlanner: unknown action '%s' — skipping", action)
                continue

            skill_entry = dict(self.ACTION_TO_SKILL[action])  # copy
            skill_entry["target"] = target
            skill_entry["reason"] = reason

            skills.append(skill_entry)
            logger.debug("TaskPlanner: %s → %s (conf_req=%.2f)",
                         action, skill_entry["skill"], skill_entry["confidence_required"])

        if not skills:
            logger.warning("TaskPlanner: produced zero skills — defaulting to scan")
            skills = [{"skill": "scan_environment", "target": "environment",
                       "reason": "no valid tasks", "confidence_required": 0.0}]

        return skills
