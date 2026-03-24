import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillLibrary:
    """
    Maps skill names to ordered lists of low-level robot actions.
    Each action string is the ROS2 topic/service name or command key
    that executor_interface will send to the Raspberry Pi.
    """

    # skill_name → ordered list of low-level action commands
    SKILLS: Dict[str, List[str]] = {
        "scan_environment": [
            "head_rotate_left",
            "head_rotate_center",
            "head_rotate_right",
            "head_tilt_down",
            "head_tilt_center",
        ],
        "navigate_to": [
            "move_base",
        ],
        "pick_object": [
            "arm_extend",
            "align_gripper",
            "close_gripper",
            "arm_retract",
        ],
        "place_object": [
            "arm_extend",
            "move_arm_to_target",
            "open_gripper",
            "arm_retract",
        ],
        "ask_user": [
            "speak_question",
            "wait_for_response",
        ],
        "wait": [
            "idle",
        ],
        "release_object": [
            "open_gripper",
            "arm_retract",
        ],
        "home_position": [
            "arm_home",
            "head_center",
        ],
    }

    def __init__(self):
        self.skills = self.SKILLS

    def get_actions(self, skill_name: str) -> List[str]:
        """Return ordered list of actions for a skill. Empty list if unknown."""
        actions = self.skills.get(skill_name, [])
        if not actions:
            logger.warning("SkillLibrary: unknown skill '%s'", skill_name)
        return actions

    def register_skill(self, skill_name: str, actions: List[str]) -> None:
        """Dynamically add or override a skill at runtime."""
        self.skills[skill_name] = actions
        logger.info("SkillLibrary: registered skill '%s' with %d actions", skill_name, len(actions))

    def list_skills(self) -> List[str]:
        return list(self.skills.keys())

    def has_skill(self, skill_name: str) -> bool:
        return skill_name in self.skills
