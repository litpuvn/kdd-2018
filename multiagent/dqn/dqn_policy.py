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


class DQNPolicy:
    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.1
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
        a_n = []
        for i in range(self.agent_count):
            state_n += (env.HEIGHT, env.WIDTH)
            a_n.append(self.action_count)

        q_state = state_n + tuple(a_n)
        DQNPolicy.Q_TABLE = np.zeros(q_state, dtype=np.float)

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

    def _build_q_state(self, state_n, action_n):
        if len(state_n) != len(action_n):
            raise Exception('Invalid length of state and action')

        state = ()
        for i in range(len(state_n)):
            state += tuple(state_n[i])

        state += tuple(action_n)

        return state

    # update q function with sample <s, a, r, s'>
    def learn(self, state_n, action_n, reward_n, next_state_n):

        current_q_state = self._build_q_state(state_n, action_n)
        q_val = DQNPolicy.Q_TABLE[current_q_state]

        max_q, _ = self._get_max_q_at_state(next_state_n)
        target_q = sum(reward_n) + DQNPolicy.DISCOUNT_FACTOR * max_q

        dq_error = (target_q - q_val)
        DQNPolicy.Q_TABLE[current_q_state] += DQNPolicy.LEARNING_RATE * dq_error

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

        # if np.random.rand() < DQNPolicy.EPSILON:
        #     # take random action
        #     action = np.random.choice(self.actions)
        # else:

            # take action according to the q function table
        all_possible_agent_actions = DQNPolicy.Q_TABLE[state]
        agent_action_offset = agent_index * self.action_count
        agent_action_end_offset = agent_action_offset + self.action_count
        agent_i_actions = all_possible_agent_actions[agent_action_offset:agent_action_end_offset]

        # avoid turning back
        action = self._get_indices_at_max_value(agent_i_actions)
        agent = self.env.get_agent(agent_index)
        while self.env.turn_back(agent.get_last_action(), action):
            action = np.random.choice(self.actions)

        return action

    def _get_max_q_at_state(self, state_n):
        q_state = ()
        action_n_tmp = []
        for i in range(self.agent_count):
            state = state_n[i]
            agent_row = state[0]
            agent_col = state[1]
            t = (agent_row, agent_col)
            q_state += t

            my_actions = self.env.allowed_agent_actions(agent_row=agent_row, agent_col=agent_col, agent_id=i)
            action_n_tmp.append(my_actions)

        # combination = itertools.product(*action_n_tmp)
        actions_Qmax_allowed = []
        # find max q
        max_val = None
        for i in itertools.product(*action_n_tmp):
            state = q_state + i
            val = DQNPolicy.Q_TABLE[state]
            if max_val is None:
                max_val = val
            if val > max_val:
                max_val = val

        # get actions
        for i in itertools.product(*action_n_tmp):
            state = q_state + i
            val = DQNPolicy.Q_TABLE[state]
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
        q_val = DQNPolicy.Q_TABLE[test_state]
        if q_val != max_val:
            raise Exception('Invalid suggested action_n')

        return max_val, action_n


    def get_action_n(self, state_n, episode=1):
        action_n = []
        agent_count = len(self.env.get_agents())

        if np.random.rand() < DQNPolicy.EPSILON and episode < 5000:
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

                my_actions = self.env.allowed_agent_actions(agent_row=agent_row, agent_col=agent_col, agent_id=i)
                # validate all my_actions

                action = np.random.choice(my_actions)
                action_n.append(action)
        else:

            qmax, action_n = self._get_max_q_at_state(state_n)
            print('state_n:', state_n, '; suggested action_n:', action_n, '; q_max', qmax)

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
        for k, v in DQNPolicy.Q_TABLE.items():
            tb.append(str(k) + str(v) + "\n")

        return tb