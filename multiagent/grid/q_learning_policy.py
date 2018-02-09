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

    def __init__(self, env):

        self.env = env
        # actions = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3]
        self.action_count = len( self.actions)

        self.agent_count = len(self.env.get_agents())

        QLearningPolicy.Q_TABLE = defaultdict(lambda: list(0 for i in range(self.agent_count*self.action_count)))

        self.state_space = 1
        for i in range(self.agent_count):
            self.state_space *= (25-i)

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

        agent_next_actions = QLearningPolicy.Q_TABLE[next_state]

        for i in range(self.agent_count):
            action = action_n[i]
            reward = reward_n[i]

            agent_i_action_offset = i * self.action_count
            agent_i_next_actions = agent_next_actions[agent_i_action_offset:(agent_i_action_offset+self.action_count)]

            current_q = QLearningPolicy.Q_TABLE[state][agent_i_action_offset + action]

            # using Bellman Optimality Equation to update q function

            new_q = reward + QLearningPolicy.DISCOUNT_FACTOR * max(agent_i_next_actions)

            QLearningPolicy.Q_TABLE[state][agent_i_action_offset + action] += QLearningPolicy.LEARNING_RATE * (new_q - current_q)

            t = 1
            # self.Q_TABLE[state][action] += QLearningPolicy.LEARNING_RATE * (new_q - current_q)

    def _get_state_string(self, state_n):
        agent_count = len(self.env.get_agents())

        state = ''
        for i in range(agent_count):
            state_i = state_n[i]
            if len(state_i) != 2:
                raise Exception("Invalid sate")

            #state = str(int(state_i[0])) + 'x' + str(int(state_i[1]))
            pos = self.env.get_pos_from_coords(state_i[0], state_i[1])
            state = str(pos)

        return state

    def get_agent_action(self, agent_index, state_n):

        # state string is the pos index mapping to all scores if move left, right, up and down
        state = self._get_state_string(state_n)

        # if np.random.rand() < QLearningPolicy.EPSILON:
        #     # take random action
        #     action = np.random.choice(self.actions)
        # else:

            # take action according to the q function table
        all_possible_agent_actions = QLearningPolicy.Q_TABLE[state]
        agent_action_offset = agent_index * self.action_count
        agent_action_end_offset = agent_action_offset + self.action_count
        agent_i_actions = all_possible_agent_actions[agent_action_offset:agent_action_end_offset]

        action = self._get_indices_at_max_value(agent_i_actions)

        return action

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
        # print("state space: ", state_space_len)

        if np.random.rand() < QLearningPolicy.EPSILON and state_space_len < self.state_space:
            # take random action
            for i in range(agent_count):
                action = np.random.choice(self.actions)
                action_n.append(action)
        else:
            # take action according to the q function table
            all_possible_agent_actions = QLearningPolicy.Q_TABLE[state]
            for i in range(self.agent_count):
                agent_action_offset = i * self.action_count
                agent_action_end_offset = agent_action_offset + self.action_count
                agent_i_actions = all_possible_agent_actions[agent_action_offset:agent_action_end_offset]
                action = self._get_indices_at_max_value(agent_i_actions)
                action_n.append(action)

        return action_n

    @staticmethod
    def _get_indices_at_max_value(state_action):
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