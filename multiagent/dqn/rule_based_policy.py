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


class RuleBasedPolicy:
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

        state_n = ()
        a_n = []
        for i in range(self.agent_count):
            state_n += (env.HEIGHT, env.WIDTH)
            a_n.append(self.action_count)

        q_state = state_n + tuple(a_n)
        RuleBasedPolicy.Q_TABLE = np.zeros(q_state, dtype=np.float)

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

        self.TRAINING_EPISODE_COUNT = 20

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _build_q_state(self, state_n, action_n):
        if len(state_n) != len(action_n):
            raise Exception('Invalid length of state and action')

        state = ()
        for i in range(len(state_n)):
            state += tuple(state_n[i])

        state += tuple(action_n)

        return state

    # update q function with sample <s, a, r, s'>
    def learn(self, state_n, action_n, reward_n, next_state_n, episode=0):
        if episode > self.TRAINING_EPISODE_COUNT:
            return

        current_q_state = self._build_q_state(state_n, action_n)
        RuleBasedPolicy.Q_TABLE[current_q_state] += sum(reward_n)

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

    def _get_actions_at_state(self, state_n):
        q_state = ()
        action_n_tmp = []
        # get possible actions
        for i in range(self.agent_count):
            state = state_n[i]
            agent_row = state[0]
            agent_col = state[1]
            agent = self.env.get_agent(i)
            t = (agent_row, agent_col)
            q_state += t

            my_actions = self._get_possible_actions(agent)
            action_n_tmp.append(my_actions)

        # combination = itertools.product(*action_n_tmp)
        actions_Qmax_allowed = []
        # find max q with all possible pairs (state_n, action_n)
        max_val = None
        for i in itertools.product(*action_n_tmp):
            state = q_state + i
            val = RuleBasedPolicy.Q_TABLE[state]
            if max_val is None:
                max_val = val
            if val > max_val:
                max_val = val

        # get actions
        for i in itertools.product(*action_n_tmp):
            state = q_state + i
            val = RuleBasedPolicy.Q_TABLE[state]
            if val == max_val:
                tmp = []
                for a in i:
                    tmp.append(a)
                actions_Qmax_allowed.append(tmp)

        if len(actions_Qmax_allowed) < 1:
            raise Exception('Error action q max')

        random_action_index = random.randrange(0, len(actions_Qmax_allowed))
        action_n = actions_Qmax_allowed[random_action_index]

        test_state = self._build_q_state(state_n, action_n)
        q_val = RuleBasedPolicy.Q_TABLE[test_state]
        if q_val != max_val:
            raise Exception('Invalid suggested action_n')

        return max_val, action_n

    def _get_possible_actions(self, agent):

        pos = agent.get_position()
        a_r = self.env.get_row(pos)
        a_c = self.env.get_col(pos)
        # validate all my_actions
        # pick shortest distance actions:
        my_actions = self.env.allowed_agent_actions(agent_row=a_r, agent_col=a_c, agent_id=agent.get_id())

        returned_actions = []

        for my_act in my_actions:
            next_state, shift_row, shift_col = agent.perform_action(my_act, actual_move=False)

            # avoid hitting the wall
            if self.env.hit_walls(next_state[0], next_state[1]):
                continue

            returned_actions.append(my_act)

        return returned_actions

    def get_action_n(self, state_n, episode=1):
        action_n = []
        agent_count = len(self.env.get_agents())

        # random walk while trainng to build rule-table
        if episode < self.TRAINING_EPISODE_COUNT or np.random.rand() < RuleBasedPolicy.EPSILON:
            # take random action
            for i in range(agent_count):
                state = state_n[i]
                agent_row = state[0]
                agent_col = state[1]

                agent = self.env.get_agent(i)
                pos = agent.get_position()
                a_r = self.env.get_row(pos)
                a_c = self.env.get_col(pos)

                if agent_row != a_r or agent_col != a_c:
                    raise Exception('invalid order of agent')

                my_actions = self._get_possible_actions(agent)
                # validate all my_actions

                action = np.random.choice(my_actions)
                action_n.append(action)
        else:

            qmax, action_n = self._get_actions_at_state(state_n)
            # print('state_n:', state_n, '; suggested action_n:', action_n, '; q_max', qmax)

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
        for k, v in RuleBasedPolicy.Q_TABLE.items():
            tb.append(str(k) + str(v) + "\n")

        return tb