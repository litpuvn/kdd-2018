import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict

class QLearningPolicy:
    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9

    EPSILON = 0.1
    Q_TABLE = None

    def __init__(self, env, actions):

        self.env = env
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.action_count = len(actions)

        self.agent_count = len(self.env.get_agents())

        default_agent_q_table = []
        for i in range(self.agent_count):
            q_values = np.zeros(len(actions))
            default_agent_q_table.append(q_values)

        QLearningPolicy.Q_TABLE = defaultdict(lambda: default_agent_q_table)

        self.state_space = 1
        for i in range(self.agent_count):
            self.state_space *= (25-i)

        a = 1

    # def build_q_table(self):
    #     for pos in range(25):
    #         for act in range(self.action_count):
    #             new_pos = 1
    #             reward = 1
    #             for p2 in range(25):
    #                 for act2 in range(self.action_count):
    #                     new_pos_2 = 1
    #                     reward_2 = 1



    # update q function with sample <s, a, r, s'>
    def learn(self, state_n, action_n, reward_n, next_state_n):

        state = self._get_state_string(state_n)

        next_state = self._get_state_string(next_state_n)

        all_agent_actions = QLearningPolicy.Q_TABLE[state]
        all_next_agent_actions = QLearningPolicy.Q_TABLE[next_state]

        for i in range(self.agent_count):
            action = action_n[i]
            reward = reward_n[i]
            single_agent_actions = all_agent_actions[i]
            current_q = single_agent_actions[action]
            next_single_agent_actions = all_next_agent_actions[i]
            # using Bellman Optimality Equation to update q function

            new_q = reward + QLearningPolicy.DISCOUNT_FACTOR * max(next_single_agent_actions)

            QLearningPolicy.Q_TABLE[state][i][action] += QLearningPolicy.LEARNING_RATE * (new_q - current_q)
            # self.Q_TABLE[state][action] += QLearningPolicy.LEARNING_RATE * (new_q - current_q)

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

        # # take action according to the q function table
        # all_possible_agent_actions = QLearningPolicy.Q_TABLE[state]
        # for i in range(self.agent_count):
        #     action = self._arg_max(all_possible_agent_actions[i])
        #     action_n.append(action)

        state_space_len = len(QLearningPolicy.Q_TABLE)
        print("state space: ", state_space_len)

        if np.random.rand() < QLearningPolicy.EPSILON and state_space_len < self.state_space:
            # take random action
            for i in range(agent_count):
                action = np.random.choice(self.actions)
                action_n.append(action)
        else:
            # take action according to the q function table
            all_possible_agent_actions = QLearningPolicy.Q_TABLE[state]
            for i in range(self.agent_count):
                action = self._arg_max(all_possible_agent_actions[i])
                action_n.append(action)

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