import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ContextBuilder:

    def __init__(self, world_model, perception=None, experience_memory=None):
        """
        Args:
            world_model:        provides get_state() → {objects, positions, relations}
            perception:         optional — provides get_current_detections() for fresh scan
            experience_memory:  optional — enriches context with past experience for this goal
        """
        self.world_model = world_model
        self.perception = perception
        self.experience_memory = experience_memory

    def build(self, goal: Optional[str] = None) -> Dict[str, Any]:
        """
        Build full context dict for the LLM planner.

        Includes:
          - world model state (objects, positions, relations)
          - fresh perception detections (if perception module connected)
          - past experience for this goal (if experience_memory connected)
          - goal target hint extracted from goal string
        """
        context: Dict[str, Any] = {}

        # 1 — World model snapshot
        try:
            state = self.world_model.get_state()
            context["objects"] = state.get("objects", [])
            context["positions"] = state.get("positions", {})
            context["relations"] = state.get("relations", [])
        except Exception as e:
            logger.error("ContextBuilder: world_model.get_state() failed (%s)", e)
            context["objects"] = []
            context["positions"] = {}
            context["relations"] = []

        # 2 — Fresh perception detections (if available)
        if self.perception is not None:
            try:
                detections = self.perception.get_current_detections()
                context["live_detections"] = detections
                logger.debug("ContextBuilder: %d live detections added", len(detections))
            except Exception as e:
                logger.warning("ContextBuilder: perception unavailable (%s)", e)
                context["live_detections"] = []
        else:
            context["live_detections"] = []

        # 3 — Past experience for this goal (if available)
        if self.experience_memory is not None and goal:
            try:
                past_plan = self.experience_memory.retrieve(goal)
                context["past_experience"] = past_plan
                if past_plan:
                    logger.debug("ContextBuilder: found past experience for goal")
            except Exception as e:
                logger.warning("ContextBuilder: experience_memory unavailable (%s)", e)
                context["past_experience"] = None
        else:
            context["past_experience"] = None

        # 4 — Summary stats useful for the LLM
        context["object_count"] = len(context["objects"])
        context["has_past_experience"] = context["past_experience"] is not None

        logger.debug(
            "ContextBuilder: built context — %d objects, %d detections, past_exp=%s",
            context["object_count"],
            len(context["live_detections"]),
            context["has_past_experience"],
        )
        return context
