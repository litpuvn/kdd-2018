import os.path
import copy
import pylab
import numpy as np
from multiagent.grid.grid_env import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

class CentralController:
    def __init__(self, env, agents, victims):

        self.env = env
        self.agents = agents
        self.agent_count = len(agents)

        self.victims = victims
        self.victim_count = len(victims)

        self.load_model = True
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)

        self.state_size = self.agent_count * 3

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self._build_model()
        self.optimizer = self._optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            fname = './save_model/reinforce_trained.h5'
            if os.path.isfile(fname):
                self.model.load_weights(fname)


    # state is input and probability of each action(policy) is output of network
    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.state_size * 3, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.state_size * 3, activation='relu'))
        model.add(Dense(self.action_size * self.agent_count, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def _optimizer(self):
        action = K.placeholder(shape=[None, self.action_size*self.agent_count])
        discounted_rewards = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
        action_prob = K.sum(action * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # calculate discounted rewards
    def _discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state_n, action_n, reward_n):

        internal_state = self._convert_to_internal_state(state_n=state_n)
        self.states.append(internal_state)

        internal_reward = self._convert_to_internal_reward(reward_n=reward_n)
        self.rewards.append(internal_reward)

        internal_action = self._convert_to_internal_action(action_n=action_n)
        self.actions.append(internal_action)

    # update policy neural network
    def train_model(self):
        discounted_rewards = np.float32(self._discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state_n):
        internal_state = self._convert_to_internal_state(state_n=state_n)

        internal_state = np.reshape(internal_state, [1, self.state_size])

        policy = self.model.predict(internal_state)[0]

        my_actions = []
        for i in range(self.agent_count):
            for j in range(self.action_size):
                my_actions.append(j)

        action_n = np.random.choice(my_actions, self.agent_count, p=policy)

        agent_action = {}
        for i in range(self.agent_count):
            agent_action[i] = action_n[i]

        return agent_action

    def _convert_to_internal_state(self, state_n):

        state = []

        for i in range(self.agent_count):
            state_i = state_n[i]
            coord_x = state_i[0]
            coord_y = state_i[1]

            state.append(coord_x)
            state.append(coord_y)

            reward = self._check_for_reward(coord_x, coord_y)
            state.append(reward)

        return state

    def _check_for_reward(self, coord_x, coord_y):

        row = self.env.get_row_from_coord(coord_y)
        col = self.env.get_col_from_coord(coord_x)

        pos = self.env.get_pos_from_row_and_col(row, col)

        reward = 0
        for v in self.victims:
            if pos == v.get_position():
                reward = reward + 1

        return reward

    def _convert_to_internal_reward(self, reward_n):

        return np.sum(reward_n)

    def _convert_to_internal_action(self, action_n):

        internal_action = []

        for i in range(self.agent_count):
            act = np.zeros(self.action_size)
            agent_action = action_n[i]
            act[agent_action] = 1
            for j in act:
                internal_action.append(j)

        return internal_action
