import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict
from multiagent.grid.base_agent import BaseAgent


class GreedyAgent(BaseAgent):

    def __init__(self, actions, agent_id, env, options):

        super().__init__(agent_id, env, options)
        # actions = [0, 1, 2, 3, 4]
        # 4 represents stays still
        self.actions = [0, 1, 2, 3, 4]

    def get_action(self, state):

        # find closest victims here
        v = self.env.get_closest_victim(self, unrescued_victims_only=not self.is_distributed())

        if v is None:
            action = 4 # stay still
        elif self.get_position() == v.get_position():
            action = 4 # stay still
        else:
            action = self.env.action_to_go_to_victim(self, v)

        return action

