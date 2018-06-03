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
import itertools


class ValueIterationPolicy:
    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.2
    DISCOUNT_FACTOR = 0.9

    EPSILON = 0.1
    Q_TABLE = None

    def __init__(self, env, info):

        self.env = env
        # actions = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3]
        self.action_count = len( self.actions)

        self.agent_count = len(self.env.get_agents())

        # if os.path.isfile('q_table_bk_2.log'):
        #     DQNPolicy.Q_TABLE = Util.read_q_table('q_table_bk.log')
        # else:
        #     DQNPolicy.Q_TABLE = defaultdict(lambda: list(0 for i in range(self.action_count**self.agent_count)))

        state_n = ()
        for i in range(self.agent_count):
            state_n += (env.HEIGHT, env.WIDTH)

        q_state = state_n
        ValueIterationPolicy.Q_TABLE = np.zeros(q_state, dtype=np.float)

        # Training Parameters
        self.brain_info = info["brain"]
        self.env_info = info["env"]

        # Learning parameters
        self.gamma = self.brain_info["discount"]
        self.learning_rate = self.brain_info["learning_rate"]

        # self._value_iteration()

    def _create_state_space(self):

        state_n = []
        for i in range(self.agent_count):
            rows = []
            for r in range(self.env.HEIGHT):
                rows.append(r)

            cols = []
            for c in range(self.env.WIDTH):
                cols.append(c)
            state_n.append(rows)
            state_n.append(cols)

        state_space = itertools.product(*state_n)

        return state_space

    def _get_state_n_tuple(self, state_n):
        state_n_tuple = ()
        for state in state_n:
            state_n_tuple += tuple(state)

        return state_n_tuple

    def _get_value(self, state_n):
        state_n_tuple = self._get_state_n_tuple(state_n)

        return ValueIterationPolicy.Q_TABLE[state_n_tuple]

    def _get_possible_action_space(self, state_n):
        action_n_tmp = []
        for i in range(self.agent_count):
            state = state_n[i]
            agent_row = state[0]
            agent_col = state[1]
            my_actions = self.env.allowed_agent_actions(agent_row=agent_row, agent_col=agent_col, agent_id=i)
            action_n_tmp.append(my_actions)

        return itertools.product(*action_n_tmp)

    def learn(self, state_n, action_n, reward_n, next_state_n, episode=1):
        if episode > 10:
            return

        next_value = self._get_value(next_state_n)
        q_value = sum(reward_n) + self.DISCOUNT_FACTOR * next_value

        state_n_tuple = self._get_state_n_tuple(state_n)
        ValueIterationPolicy.Q_TABLE[state_n_tuple] += q_value

    def _value_iteration(self):

        for i in range(10000):
            state_space = self._create_state_space()
            for state_n_tuple in state_space:
                state_n = self._get_state_n_from_tuple(state_n_tuple)
                value_list = []
                for action_n_tuple in self._get_possible_action_space(state_n):
                    action_n = self._get_action_n_from_tuple(action_n_tuple)

                    next_state_n = []
                    reward_n = []
                    for i in range(self.agent_count):
                        agent = self.env.get_agent(i)
                        next_state, _, _ = agent.perform_action(action_n[i], actual_move=False)
                        reward = self.env.get_reward_at_pos(next_state[0], next_state[1])

                        next_state_n.append(next_state)
                        reward_n.append(reward)

                    next_value = self._get_value(next_state_n)
                    q_value = sum(reward_n) + self.DISCOUNT_FACTOR*next_value
                    value_list.append(q_value)

                max_val = max(value_list)
                ValueIterationPolicy.Q_TABLE[state_n_tuple] += max_val

        return ValueIterationPolicy.Q_TABLE

    def _get_state_n_from_tuple(self, state_n_tuple):
        state_n = []
        for i in range(0, len(state_n_tuple) - 1, 2):
            state_n.append([state_n_tuple[i], state_n_tuple[i + 1]])

        return state_n

    def _get_action_n_from_tuple(self, action_n_tuple):
        action_n = []
        for i in range(len(action_n_tuple)):
            action_n.append(action_n_tuple[i])

        return action_n


    # # get action according to the current value function table
    # def get_action(self, state):
    #     action_list = []
    #     max_value = -99999
    #
    #     if state == [2, 2]:
    #         return []
    #
    #     # calculating q values for the all actions and
    #     # append the action to action list which has maximum q value
    #     for action in self.env.possible_actions:
    #
    #         next_state = self.env.state_after_action(state, action)
    #         reward = self.env.get_reward(state, action)
    #         next_value = self.get_value(next_state)
    #         value = (reward + self.discount_factor * next_value)
    #
    #         if value > max_value:
    #             action_list.clear()
    #             action_list.append(action)
    #             max_value = value
    #         elif value == max_value:
    #             action_list.append(action)
    #
    #     return action_list

    def get_action_n(self, state_n, episode=1):
        action_list = []
        max_value = -9999999

        for action_n_tuple in self._get_possible_action_space(state_n):
            action_n = self._get_action_n_from_tuple(action_n_tuple)
            next_state_n = []
            reward_n = []
            for i in range(self.agent_count):
                agent = self.env.get_agent(i)
                next_state, _, _ = agent.perform_action(action_n[i], actual_move=False)
                reward = self.env.get_reward_at_pos(next_state[0], next_state[1])

                next_state_n.append(next_state)
                reward_n.append(reward)

            next_value = self._get_value(next_state_n)
            q_value = sum(reward_n) + self.DISCOUNT_FACTOR * next_value

            if q_value > max_value:
                action_list.clear()
                action_list.append(action_n)
                max_value = q_value
            elif q_value == max_value:
                action_list.append(action_n)

        return action_list[np.random.choice(len(action_list))]

