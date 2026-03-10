class ExecutorInterface:

    def __init__(self):

        pass

    def send(self, action):

        print("Robot Action:", action.action, action.params)