import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict
from multiagent.grid.base_agent import BaseAgent


class GreedyAgent(BaseAgent):

    def __init__(self, actions, agent_id):

        super().__init__(agent_id)
        # actions = [0, 1, 2, 3, 4]
        # 4 represents stays still
        self.actions = [0, 1, 2, 3, 4]

    def get_action(self, state):

        # find closest victims here
        action = np.random.choice(self.actions)

        return action

