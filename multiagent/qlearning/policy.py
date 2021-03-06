import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict
import os.path
from multiagent.grid.util import Util
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D
from keras.models import load_model
from collections import deque


class Policy:
    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9

    EPSILON = 0.1
    Q_TABLE = None

    def __init__(self, env, info):

        self.env = env
        # actions = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3]
        self.action_count = len( self.actions)

        self.agent_count = len(self.env.get_agents())

        if os.path.isfile('q_table_bk_2.log'):
            Policy.Q_TABLE = Util.read_q_table('q_table_bk.log')
        else:
            Policy.Q_TABLE = defaultdict(lambda: list(0 for i in range(self.action_count**self.agent_count)))

        self.state_space = 1
        for i in range(self.agent_count):
            self.state_space *= (25-i)

        self.memory = deque(maxlen=100000)

        # Training Parameters
        self.brain_info = info["brain"]
        self.env_info = info["env"]

        # Learning parameters
        self.gamma = self.brain_info["discount"]
        self.learning_rate = self.brain_info["learning_rate"]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # update q function with sample <s, a, r, s'>
    def learn(self, state_n, action_n, reward_n, next_state_n):

        state = self._get_state_string(state_n)

        next_state = self._get_state_string(next_state_n)

        # agent_next_actions = Policy.Q_TABLE[next_state]

        n_action_index = self._get_n_actions_index(action_n)
        current_q = Policy.Q_TABLE[state][n_action_index]

        next_possible_q_values = Policy.Q_TABLE[next_state]
        new_q = np.sum(reward_n) + Policy.DISCOUNT_FACTOR * max(next_possible_q_values)
        Policy.Q_TABLE[state][n_action_index] += Policy.LEARNING_RATE * (new_q - current_q)

        # if len(Policy.Q_TABLE) > 625:
        #     i = 0
        #     raise Exception('Something wrong with Q_table')

        # for i in range(self.agent_count):
        #     action = action_n[i]
        #     reward = reward_n[i]
        #
        #     agent_i_action_offset = i * self.action_count
        #     agent_i_next_actions = agent_next_actions[agent_i_action_offset:(agent_i_action_offset+self.action_count)]
        #
        #     current_q = Policy.Q_TABLE[state][agent_i_action_offset + action]
        #
        #     # using Bellman Optimality Equation to update q function
        #
        #     new_q = reward + Policy.DISCOUNT_FACTOR * max(agent_i_next_actions)
        #
        #     Policy.Q_TABLE[state][agent_i_action_offset + action] += Policy.LEARNING_RATE * (new_q - current_q)
        #
        #     t = 1
            # self.Q_TABLE[state][action] += Policy.LEARNING_RATE * (new_q - current_q)

    def _get_n_actions_index(self, action_n):
        binary_string = ''
        for action in action_n:
            action_bin = bin(action)
            action_bin = action_bin.lstrip('0b')

            if len(action_bin) == 0:
                action_bin = '00'

            if len(action_bin) % 2 == 1:
                action_bin = '0' + action_bin

            binary_string = action_bin + binary_string

        return int(binary_string, 2)

    def _get_state_string(self, state_n):
        agent_count = len(self.env.get_agents())

        state = ''
        for i in range(agent_count):
            state_i = state_n[i]
            if len(state_i) != 2:
                raise Exception("Invalid sate")

            pos = '[' + str(int(state_i[0])) + ', ' + str(int(state_i[1])) + ']'
            state = state + pos

        return state

    def get_agent_action(self, agent_index, state_n):

        # state string is the pos index mapping to all scores if move left, right, up and down
        state = self._get_state_string(state_n)

        # if np.random.rand() < Policy.EPSILON:
        #     # take random action
        #     action = np.random.choice(self.actions)
        # else:

            # take action according to the q function table
        all_possible_agent_actions = Policy.Q_TABLE[state]
        agent_action_offset = agent_index * self.action_count
        agent_action_end_offset = agent_action_offset + self.action_count
        agent_i_actions = all_possible_agent_actions[agent_action_offset:agent_action_end_offset]

        # avoid turning back
        action = self._get_indices_at_max_value(agent_i_actions)
        agent = self.env.get_agent(agent_index)
        while self.env.turn_back(agent.get_last_action(), action):
            action = np.random.choice(self.actions)

        return action

    def get_action_n(self, state_n):
        action_n = []
        agent_count = len(self.env.get_agents())

        state = self._get_state_string(state_n)

        # # take action according to the q function table
        # all_possible_agent_actions = Policy.Q_TABLE[state]
        # for i in range(self.agent_count):
        #     action = self._arg_max(all_possible_agent_actions[i])
        #     action_n.append(action)

        state_space_len = len(Policy.Q_TABLE)
        # print("state space: ", state_space_len)

        # if np.random.rand() < Policy.EPSILON:
        #     # take random action
        #     for i in range(agent_count):
        #         action = np.random.choice(self.actions)
        #         action_n.append(action)
        # else:
        # take action according to the q function table
        all_possible_agent_actions = Policy.Q_TABLE[state]
        n_action_index = self._get_indices_at_max_value(all_possible_agent_actions)
        n_action_bin = bin(n_action_index)
        n_action_bin = n_action_bin.lstrip("0b")

        missing_char_count = 2**self.agent_count - len(n_action_bin)
        for i in range(missing_char_count):
            n_action_bin = '0' + n_action_bin

        actions = []
        for i in range(0, len(n_action_bin), 2):
            action_i = n_action_bin[i:(i+2)]
            action_i = int(action_i, 2)
            actions.append(action_i)

        actions = list(reversed(actions))
        for i in range(len(actions)):
            agent = self.env.get_agent(i)
            action_i = actions[i]
            while self.env.turn_back(agent.get_last_action(), action_i):
                action_i = np.random.choice(self.actions)

            action_n.append(action_i)

        # for i in range(self.agent_count):
        #     agent_action_offset = i * self.action_count
        #     agent_action_end_offset = agent_action_offset + self.action_count
        #     agent_i_actions = all_possible_agent_actions[agent_action_offset:agent_action_end_offset]
        #     action = self._get_indices_at_max_value(agent_i_actions)
        #     action_n.append(action)

        # if np.random.rand() < Policy.EPSILON and state_space_len < self.state_space:
        #     # take random action
        #     for i in range(agent_count):
        #         action = np.random.choice(self.actions)
        #         action_n.append(action)
        # else:
        #     # take action according to the q function table
        #     all_possible_agent_actions = Policy.Q_TABLE[state]
        #     for i in range(self.agent_count):
        #         agent_action_offset = i * self.action_count
        #         agent_action_end_offset = agent_action_offset + self.action_count
        #         agent_i_actions = all_possible_agent_actions[agent_action_offset:agent_action_end_offset]
        #         action = self._get_indices_at_max_value(agent_i_actions)
        #         action_n.append(action)

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

    def get_qtable(self):

        tb = []
        for k, v in Policy.Q_TABLE.items():
            tb.append(str(k) + str(v) + "\n")

        return tb