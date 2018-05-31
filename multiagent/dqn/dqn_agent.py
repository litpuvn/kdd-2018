import numpy as np
import random
from multiagent.grid.grid_env import Env
from collections import defaultdict
from multiagent.grid.base_agent import BaseAgent

from collections import deque

class DQNAgent(BaseAgent):

    # Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    LEARNING_RATE = 0.01
    DISCOUNT_FACTOR = 0.9

    EPSILON = 0.1

    def __init__(self, actions, agent_id, env, options):

        super().__init__(agent_id, env, options=options)

        # actions = [0, 1, 2, 3]
        self.actions = [0, 1, 2, 3, 4]

        self.Q_TABLE = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

        # self.action_dict = {"up": 0, "down": 1, "left": 2, "right": 3}
        self.action_coords = np.array([[-1,0],  [1,0], [0,-1], [0,1]], dtype=np.int)

        self.action_history = []

        self.targets = None

    def _step_distance(self, a_r, a_c, v_r, v_c):
        return abs(a_r - v_r) + abs(a_c - v_c)

    def setup_targets_order(self):
        if self.targets is not None:
            return

        pos = self.get_position()
        a_r = self.env.get_row(pos)
        a_c = self.env.get_col(pos)

        env = self.env

        def target_distance(v):
            pos = v.get_position()
            v1_r = env.get_row(pos)
            v1_c = env.get_col(pos)
            d1 = abs(a_r - v1_r) + abs(a_c - v1_c)
            return d1

        my_victims = []
        for v in self.env.victims:
            if v.get_reward() <= 0:
                continue

            my_victims.append(v)

        my_victims.sort(key=target_distance)
        self.targets = my_victims

    def pick_target(self):
        for i in range(len(self.targets)):
            v = self.targets[i]

            if not v.is_rescued():
                return v

        raise Exception('Not found v')
    # ========================
    # Action utilities - jumping 1 cell
    # ========================
    def perform_action(self, action, actual_move=True):
        pos = self.get_position()
        current_row = self.env.get_row(pos)
        current_col = self.env.get_col(pos)

        state = [current_row, current_col]
        next_state = np.add(state, self.action_coords[action])

        # if self.env.hit_walls(next_state[0], next_state[1]):
        #     raise Exception('invalid position, agent=' + str(self.get_id()))

        shift_row = next_state[0] - current_row
        shift_col = next_state[1] - current_col

        new_pos = self.env.get_pos_from_row_and_col(next_state[0], next_state[1])

        if actual_move == True:
            self.set_position(new_pos)
            self.action_history.append({'pos': pos, 'action': action})

        return next_state, shift_row, shift_col

    def reset(self):
        self.reset_rescued_victims()
        self.reset_last_action()
        self.reset_history()
        self.targets = None
        self.setup_targets_order()

    def get_action_history(self):
        return self.action_history

    def reset_history(self):
        self.action_history =[]
    # # update q function with sample <s, a, r, s'>
    # def learn(self, state, action, reward, next_state):
    #     current_q = self.Q_TABLE[state][action]
    #     # using Bellman Optimality Equation to update q function
    #     new_q = reward + QLearningAgent.DISCOUNT_FACTOR * max(self.Q_TABLE[next_state])
    #     self.Q_TABLE[state][action] += QLearningAgent.LEARNING_RATE * (new_q - current_q)
    #
    #     b = 1

    # get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy
    # def get_action(self, state):
    #     if np.random.rand() < QLearningAgent.EPSILON:
    #         # take random action
    #         action = np.random.choice(self.actions)
    #     else:
    #         # take action according to the q function table
    #         state_action = self.Q_TABLE[state]
    #         action = self._arg_max(state_action)
    #     return action

    # get list of indices that have max values
    # then return a random index in this max_index_list
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

        ra = random.choice(max_index_list)
        return ra
