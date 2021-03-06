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
from heapq import heappop, heappush


class HQPolicy:
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
        a_n = []
        for i in range(self.agent_count):
            state_n += (env.HEIGHT, env.WIDTH)
            a_n.append(self.action_count)

        q_state = state_n + tuple(a_n)
        HQPolicy.Q_TABLE = np.zeros(q_state, dtype=np.float)

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
        # self.heuristic_table = self._build_heuristic_table(q_state)
        self.heuristic_table = np.zeros(q_state, dtype=np.float)

        ## init q table
        # HQPolicy.Q_TABLE = np.copy(-1*self.heuristic_table)

    def _heuristic(self, cell, goal):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

        # find path should be from initial pos

    def _find_path(self, agent, victim, agent_from_pos):
        pos = agent_from_pos
        a_r = self.env.get_row(pos)
        a_c = self.env.get_col(pos)

        pos = victim.get_position()
        v_r = self.env.get_row(pos)
        v_c = self.env.get_col(pos)

        start, goal = (a_r, a_c), (v_r, v_c)

        pr_queue = []
        heappush(pr_queue, (0 + self._heuristic(start, goal), 0, "", start))

        graph = self.graph

        visited = set()
        while pr_queue:
            _, cost, path, current = heappop(pr_queue)
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
                heappush(pr_queue, (cost + self._heuristic(neighbour, goal), cost + 1,
                                    path + direction, neighbour))

    # def _get_heuristics_at_pos(self, row, col):
    #
    #     h_min = 9999999
    #     for v in self.env.victims:
    #         pos = v.get_position()
    #         r = self.env.get_row(pos)
    #         c = self.env.get_col(pos)
    #
    #         if v.get_reward() < 0:
    #             if r == row and c == col:
    #                 return 10000
    #             continue
    #
    #         h = self._heuristic([r, c], [row, col])
    #         if h < h_min:
    #             h_min = h
    #
    #     return h_min

    def _build_heuristic_table(self, state_action_space):
        table = np.zeros(state_action_space, dtype=np.float)
        state_space = self._create_state_space()
        for state_n_tuple in state_space:
            state_n = self._get_state_n_from_tuple(state_n_tuple)
            for action_n_tuple in self._get_possible_action_space(state_n):
                action_n = self._get_action_n_from_tuple(action_n_tuple)

                next_state_n = []
                for i in range(len(state_n)):
                    state = state_n[i]
                    agent = self.env.get_agent(i)
                    next_state, _, _ = agent.perform_action(action_n[i], actual_move=False, from_state=state)
                    next_state_n.append(next_state)

                h_value, _ = self._get_heuristics_at_state(state_n)
                table[state_n_tuple + action_n_tuple] += h_value

        return table

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

    def _get_state_n_tuple(self, state_n):
        state_n_tuple = ()
        for state in state_n:
            state_n_tuple += tuple(state)

        return state_n_tuple

    def _get_value(self, state_n):
        state_n_tuple = self._get_state_n_tuple(state_n)

        return self.heuristic_table[state_n_tuple]

    def _get_possible_action_space(self, state_n):
        action_n_tmp = []
        for i in range(self.agent_count):
            state = state_n[i]
            agent_row = state[0]
            agent_col = state[1]
            my_actions = self.env.allowed_agent_actions(agent_row=agent_row, agent_col=agent_col, agent_id=i)
            action_n_tmp.append(my_actions)

        return itertools.product(*action_n_tmp)

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
        q_val = HQPolicy.Q_TABLE[current_q_state]

        max_q, _ = self._get_max_q_at_state(next_state_n, True)
        target_q = sum(reward_n) + HQPolicy.DISCOUNT_FACTOR * max_q

        dq_error = (target_q - q_val)
        HQPolicy.Q_TABLE[current_q_state] += HQPolicy.LEARNING_RATE * dq_error

    def update_heuristics(self, state_n, action_n, reward_n, next_state_n):

        current_q_state = self._build_q_state(state_n, action_n)
        h_val = self.heuristic_table[current_q_state]

        h_pre_cost, _ = self._get_heuristics_at_state(state_n)
        h_next_cost, _ = self._get_heuristics_at_state(next_state_n)

        h_cost = h_next_cost - h_pre_cost
        # h_cost = h_next_cost

        target_h = sum(reward_n) + HQPolicy.DISCOUNT_FACTOR * (10-h_cost)

        dq_error = (target_h - h_val)
        self.heuristic_table[current_q_state] += HQPolicy.LEARNING_RATE * dq_error

    def _get_heuristics_at_state(self, state_n):

        dtype = [('state', int), ('victime', int), ('distance', float)]
        values = []

        for i in range(len(state_n)):
            state_i = state_n[i]
            for v in self.env.victims:
                if v.get_reward() < 0 or v.is_rescued():
                    continue
                pos = v.get_position()
                v_r = self.env.get_row(pos)
                v_c = self.env.get_col(pos)
                v_state = [v_r, v_c]
                distance = self._heuristic(state_i, v_state)
                values.append((i, v.get_id(), distance))

        np_values = np.array(values, dtype=dtype)
        sorted_values = np.sort(np_values, order=['distance'])

        total_distance = 0
        catched_a_i = []
        catched_v_i = []

        matched = []
        for i in range(len(sorted_values)):
            a_i, v_i, d = sorted_values[i]

            if a_i in catched_a_i or v_i in catched_v_i:
                continue

            catched_a_i.append(a_i)
            catched_v_i.append(v_i)
            matched.append((a_i, v_i, d))
            total_distance += d

        return total_distance, matched


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

    def _get_combined_value_at_state(self, state_action):

        q_val = HQPolicy.Q_TABLE[state_action]
        h_val = self.heuristic_table[state_action]

        val = q_val + h_val

        return val

    def _get_max_q_at_state(self, state_n, learning=False, greedy_selection=False):
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
        for a in itertools.product(*action_n_tmp):
            state = q_state + a
            val = HQPolicy.Q_TABLE[state]

            if max_val is None:
                max_val = val
            else:
                if val > max_val:
                    max_val = val

        # get actions
        for i in itertools.product(*action_n_tmp):
            state = q_state + i
            val = HQPolicy.Q_TABLE[state]
            if val == max_val:
                tmp = []
                for a in i:
                    tmp.append(a)
                actions_Qmax_allowed.append(tmp)

        if len(actions_Qmax_allowed) < 1:
            raise Exception('Error action q max')

        if greedy_selection == False:
            random_action_index = random.randrange(0, len(actions_Qmax_allowed))
            action_n = actions_Qmax_allowed[random_action_index]
        else:
            h_min = 999999
            action_n = []
            min_h_actions = []
            for a_n in actions_Qmax_allowed:
                next_state_n = []
                for i in range(len(a_n)):
                    agent = self.env.get_agent(i)
                    action = a_n[i]
                    next_state, _, _ = agent.perform_action(action, actual_move=False)
                    next_state_n.append(next_state)

                h_cost, _ = self._get_heuristics_at_state(next_state_n)
                if h_cost < h_min:
                    h_min = h_cost
                    min_h_actions.clear()
                    min_h_actions.append(a_n)
                elif h_cost == h_min:
                    min_h_actions.append(a_n)

            random_action_index = random.randrange(0, len(min_h_actions))
            action_n = min_h_actions[random_action_index]

        test_state = self._build_q_state(state_n, action_n)
        q_val = HQPolicy.Q_TABLE[test_state]
        if q_val != max_val:
            raise Exception('Invalid suggested action_n')

        return max_val, action_n

    def _get_actions_with_max_combined_value(self, state_n):
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
        for a in itertools.product(*action_n_tmp):
            state = q_state + a
            val = self._get_combined_value_at_state(state)
            state_n = self._get_state_n_from_tuple(q_state)
            action_n = self._get_action_n_from_tuple(a)
            next_state_n = []
            for i in range(len(state_n)):
                agent_i = self.env.get_agent(i)
                action = action_n[i]
                next_state, _, _ = agent_i.perform_action(action, actual_move=False, from_state=state_n[i])
                next_state_n.append(next_state)

            h_cost_next, _ = self._get_heuristics_at_state(next_state_n)
            h_cost_pre, _ = self._get_heuristics_at_state(state_n)
            dh_error = -h_cost_next + h_cost_pre
            val += HQPolicy.LEARNING_RATE * dh_error
            if max_val is None:
                max_val = val
                actions_Qmax_allowed.append(action_n)
            else:
                if val > max_val:
                    actions_Qmax_allowed.clear()
                    max_val = val
                    actions_Qmax_allowed.append(action_n)
                elif val == max_val:
                    actions_Qmax_allowed.append(action_n)


        # actions_Qmax_allowed.append(a_n_max)
        # get actions
        # for i in itertools.product(*action_n_tmp):
        #     state = q_state + i
        #     val = self._get_combined_value_at_state(state)
        #     if val == max_val:
        #         tmp = []
        #         for a in i:
        #             tmp.append(a)
        #         actions_Qmax_allowed.append(tmp)

        if len(actions_Qmax_allowed) < 1:
            raise Exception('Error action q max')

        random_action_index = random.randrange(0, len(actions_Qmax_allowed))
        action_n = actions_Qmax_allowed[random_action_index]

        # test_state = self._build_q_state(state_n, action_n)
        # q_val = self._get_combined_value_at_state(test_state)
        # if q_val != max_val:
        #     raise Exception('Invalid suggested action_n')

        return max_val, action_n

    def get_action_n(self, state_n, episode=1):

        action_n = []
        agent_count = len(self.env.get_agents())

        if np.random.rand() < HQPolicy.EPSILON and episode < 0:
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

            qmax, action_n = self._get_max_q_at_state(state_n, learning=False, greedy_selection=True)
            # qmax, action_n = self._get_actions_with_max_combined_value(state_n)
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
        for k, v in HQPolicy.Q_TABLE.items():
            tb.append(str(k) + str(v) + "\n")

        return tb

    def get_htable(self):
        return self.heuristic_table