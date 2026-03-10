from pydantic import BaseModel
from typing import List, Dict


class Goal(BaseModel):
    goal: str
    target: str | None = None
    constraints: List[str] = []


class Task(BaseModel):
    task: str
    params: Dict = {}


class Action(BaseModel):
    action: str
    params: Dict = {}