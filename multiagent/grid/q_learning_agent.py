import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict
from multiagent.grid.base_agent import BaseAgent


class QLearningAgent(BaseAgent):

    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9

    EPSILON = 0.1

    def __init__(self, actions, agent_id, env):

        super().__init__(agent_id, env)

        # actions = [0, 1, 2, 3]
        self.actions = actions

        self.Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.Q_TABLE[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + QLearningAgent.DISCOUNT_FACTOR * max(self.Q_TABLE[next_state])
        self.Q_TABLE[state][action] += QLearningAgent.LEARNING_RATE * (new_q - current_q)

        b = 1

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() < QLearningAgent.EPSILON:
            # take random action
            action = np.random.choice(self.actions)
        else:
            # take action according to the q function table
            state_action = self.Q_TABLE[state]
            action = self._arg_max(state_action)
        return action

    # get list of indices that have max values
    # then return a random index in this max_index_list
    @staticmethod
    def _arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)

        ra = random.choice(max_index_list)
        return ra
