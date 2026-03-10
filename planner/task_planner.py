import json
from planner.schemas import Task


class TaskPlanner:

    def __init__(self, llm):

        self.llm = llm

    def plan(self, goal):

        prompt = f"""
Convert goal into ordered robot tasks.

Goal:
{goal.model_dump()}

Return JSON list:
[
 {{"task":"...", "params":{{}}}}
]
"""

        result = self.llm.generate(prompt)

        tasks = json.loads(result)

        return [Task(**t) for t in tasks]