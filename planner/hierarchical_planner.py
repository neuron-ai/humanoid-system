from planner.high_level_planner import HighLevelPlanner
from planner.task_planner import TaskPlanner
from planner.action_planner import ActionPlanner


class HierarchicalPlanner:

    def __init__(self, high_level, task_planner, action_planner, executor):

        self.high_level = high_level
        self.task_planner = task_planner
        self.action_planner = action_planner
        self.executor = executor

    def run(self, instruction, world_model):

        goal = self.high_level.plan(instruction, world_model)

        tasks = self.task_planner.plan(goal)

        for task in tasks:

            actions = self.action_planner.plan(task)

            for action in actions:

                self.executor.send(action)