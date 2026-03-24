from typing import TypedDict, List, Dict, Any, Optional


class PlannerState(TypedDict):

    # Input
    goal: str

    # Built by context_builder
    context: Dict[str, Any]

    # LLM output — list of dicts with action/reason fields
    tasks: List[Dict]

    # Skill library output — list of {skill: str} dicts
    skills: List[Dict]

    # Currently executing skill
    current_skill: Dict

    # Executor output — list of {action, status} dicts
    execution_result: List[Dict]

    # Failure tracking
    failure_reason: Optional[str]
    retry_count: int
    scan_count: int

    # Perception confidence of held object after grasp
    grasp_confidence: Optional[float]

    # Ambiguity — multiple candidates found
    candidates: Optional[List[Dict]]
    waiting_for_user: bool

    # LLM fallback flag
    llm_fallback: bool

    # Terminal flags
    done: bool
    error: Optional[str]
