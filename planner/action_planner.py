import json
from planner.schemas import Action


class ActionPlanner:

    def __init__(self, llm):

        self.llm = llm

    def plan(self, task):

        prompt = f"""
Convert robot task to executable actions.

Task:
{task.model_dump()}

Return JSON list:
[
 {{"action":"...", "params":{{}}}}
]
"""

        result = self.llm.generate(prompt)

        actions = json.loads(result)

        return [Action(**a) for a in actions]