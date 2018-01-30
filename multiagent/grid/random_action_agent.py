import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict
from multiagent.grid.base_agent import BaseAgent


class RandomActionAgent(BaseAgent):

    def __init__(self, actions, agent_id, env):

        super().__init__(agent_id, env)
        # actions = [0, 1, 2, 3]
        self.actions = actions

    def get_action(self, state):

        action = np.random.choice(self.actions)

        return action

