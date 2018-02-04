import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict

class QLearningPolicy:
    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9

    EPSILON = 0.1

    def __init__(self, env, actions):

        self.env = env
        # actions = [0, 1, 2, 3]
        self.actions = actions

        agent_count = len(self.env.get_agents())
        rewards = np.zeros(agent_count * len(actions))
        self.Q_TABLE = defaultdict(lambda: rewards)

    # update q function with sample <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.Q_TABLE[state][action]
        # using Bellman Optimality Equation to update q function
        new_q = reward + QLearningPolicy.DISCOUNT_FACTOR * max(self.Q_TABLE[next_state])
        self.Q_TABLE[state][action] += QLearningPolicy.LEARNING_RATE * (new_q - current_q)

    def _get_state_string(self, state_n):
        agent_count = len(self.env.get_agents())

        state = ''
        for i in range(agent_count):
            state_i = str(state_n[i])
            state += state_i

        return state

    def get_action_n(self, state_n):
        action_n = []
        agent_count = len(self.env.get_agents())

        state = self._get_state_string(state_n)

        if np.random.rand() < QLearningPolicy.EPSILON:
            # take random action
            for i in range(agent_count):
                action = np.random.choice(self.actions)
                action_n.append(action)
        else:
            # take action according to the q function table
            state_action = self.Q_TABLE[state]
            action_n = self._arg_max(state_action)

        return action_n

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
        return random.choice(max_index_list)