"""
 EnvironmentClass.py  (author: Anson Wong / git: ankonzoid)
"""
import numpy as np

class Environment:
    def __init__(self, env_info):
        self.name = "GridWorld"

        # Read environment parameters
        self.Ny = env_info["Ny"]  # y-grid size
        self.Nx = env_info["Nx"]  # x-grid size

        # State and Action space
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = np.array([[-1,0], [0,1], [1,0], [0,-1]], dtype=np.int)
        self.N_actions = len(self.action_dict.keys())

        self.state_dim = (self.Ny, self.Nx)
        self.action_dim = (self.N_actions,)
        self.state_action_dim = self.state_dim + self.action_dim

        # Rewards
        self.reward = self.define_rewards()

        # Make checks
        if self.N_actions != len(self.action_coords):
            raise IOError("Inconsistent actions given")

    # ========================
    # Rewards
    # ========================
    def define_rewards(self):
        R_goal = 100  # reward for arriving at a goal state
        R_nongoal = -0.1  # reward for arriving at a non-goal state
        reward = R_nongoal * np.ones(self.state_action_dim, dtype=np.float)
        reward[self.Ny-2, self.Nx-1, self.action_dict["down"]] = R_goal
        reward[self.Ny-1, self.Nx-2, self.action_dict["right"]] = R_goal
        return reward

    def get_reward(self, state, action):
        sa = tuple(list(state) + [action])
        return self.reward[sa]

    # ========================
    # Action restrictions
    # ========================
    def allowed_actions(self, state):
        actions = []
        y = state[0]
        x = state[1]
        if (y > 0): actions.append(self.action_dict["up"])
        if (y < self.Ny-1): actions.append(self.action_dict["down"])
        if (x > 0): actions.append(self.action_dict["left"])
        if (x < self.Nx-1): actions.append(self.action_dict["right"])
        actions = np.array(actions, dtype=np.int)
        return actions

    # ========================
    # Environment Details
    # ========================
    def starting_state(self):
        return np.array([0, 0], dtype=np.int)

    def is_terminal(self, state):
        return np.array_equal(state, np.array([self.Ny-1, self.Nx-1], dtype=np.int))

    # ========================
    # Action utilities - jumping 1 cell
    # ========================
    def perform_action(self, state, action):
        return np.add(state, self.action_coords[action])