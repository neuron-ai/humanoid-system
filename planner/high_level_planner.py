import json
from planner.schemas import Goal


class HighLevelPlanner:

    def __init__(self, llm, context_builder):

        self.llm = llm
        self.context_builder = context_builder

    def plan(self, instruction, world_model):

        context = self.context_builder.build(world_model)

        prompt = f"""
You are a humanoid retail robot.

Environment:
{context}

Instruction:
{instruction}

Return JSON:
{{"goal":"...", "target":"...", "constraints":[]}}
"""

        result = self.llm.generate(prompt)

        data = json.loads(result)

        return Goal(**data)