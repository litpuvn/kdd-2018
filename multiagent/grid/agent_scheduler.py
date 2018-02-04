import copy
import pylab
import numpy as np
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import os.path

EPISODES = 2500


# this is REINFORCE Agent for GridWorld
class AgentScheduler:
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
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            fname = './save_model/reinforce_trained.h5'
            if os.path.isfile(fname):
                self.model.load_weights(fname)

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def optimizer(self):
        action = K.placeholder(shape=[None, 5])
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

    # get action from policy network
    def get_action(self, state_n):

        # build state here
        internal_state = self._get_internal_state(state_n)
        internal_state = np.reshape(internal_state, [1, 15])

        policy = self.model.predict(internal_state)[0]

        return np.random.choice(self.action_size, 1, p=policy)

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy neural network
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def _get_internal_state(self, state_n):

        state = []

        for i in range(self.agent_count):
            state_i = state_n[i]
            coord_x = state_i[0]
            coord_y = state_i[1]
            for v in self.victims:
                vx = self.env.get_column_center_pixel(v.get_position())
                vy = self.env.get_row_center_pixel(v.get_position())
                state.append(vx - coord_x)
                state.append(vy - coord_y)
                # reward = self._check_for_reward(coord_x, coord_y)
                reward = v.get_reward()
                state.append(reward)

                if reward < 0:
                    state.append(-1)

        return state

    def _check_for_reward(self, coord_x, coord_y):

        row = self.env.get_row_from_coord(coord_y)
        col = self.env.get_col_from_coord(coord_x)

        pos = self.env.get_pos_from_row_and_col(row, col)

        reward = 0
        for v in self.victims:
            if pos == v.get_position():
                reward = v.get_reward()

        return reward


