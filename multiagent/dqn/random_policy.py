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


class RandomPolicy:
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
        RandomPolicy.Q_TABLE = np.zeros(q_state, dtype=np.float)

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
    # def learn(self, state_n, action_n, reward_n, next_state_n):
        # print('learning randomly')

    def get_action_n(self, state_n, episode=1):
        action_n = []
        agent_count = len(self.env.get_agents())
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

        return action_n

