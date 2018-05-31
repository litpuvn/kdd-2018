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


class BFSPolicy:
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
        BFSPolicy.Q_TABLE = np.zeros(q_state, dtype=np.float)

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

        self.graph = self.env.get_graph_representation()
        self.agent_last_target = {}

    def reset(self):
        self.agent_last_target = {}

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _step_distance(self, a_r, a_c, v_r, v_c):
        return abs(a_r - v_r) + abs(a_c - v_c)

    # find path should be from initial pos
    def _find_path(self, agent, victim, agent_from_pos):
        pos = agent_from_pos
        a_r = self.env.get_row(pos)
        a_c = self.env.get_col(pos)

        pos = victim.get_position()
        v_r = self.env.get_row(pos)
        v_c = self.env.get_col(pos)

        start, goal = (a_r, a_c), (v_r, v_c)
        graph = self.graph

        queue = deque([("", start)])
        visited = set()
        while queue:
            path, current = queue.popleft()
            if current == goal:
                total_step_taken = len(agent.get_action_history())
                if total_step_taken > len(path):
                    raise Exception('Invalid path and historical steps')

                current_step = path[total_step_taken]
                action = self.env.get_action_from_direction(current_step)
                return action, total_step_taken, path
            if current in visited:
                continue
            visited.add(current)
            for direction, neighbour in graph[current]:
                queue.append((path + direction, neighbour))

        raise Exception('No way')

    def _get_possible_actions(self, agent):

        agent_id = agent.get_id()
        target_victim = agent.pick_target()
        if agent_id not in self.agent_last_target:
            agent.set_start_target_search_position(agent.get_position())
            self.agent_last_target[agent_id] = target_victim

        agent_pos = agent.get_start_target_search_position()
        last_target_victim = self.agent_last_target[agent_id]
        if target_victim is not last_target_victim:
            agent_pos = agent.get_position()
            agent.set_start_target_search_position(agent_pos)

            self.agent_last_target[agent_id] = target_victim
            # reset history of first finding
            agent.reset_history()

        current_action, _, path = self._find_path(agent, target_victim, agent_from_pos=agent_pos)

        return [current_action]

    def get_action_n(self, state_n, episode=1, episode_time_step=1):
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

            my_actions = self._get_possible_actions(agent)
            action = np.random.choice(my_actions)
            action_n.append(action)

        return action_n

