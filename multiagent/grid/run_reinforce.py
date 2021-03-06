#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from multiagent.speaker_listener import SpeakerListenerScenario

import numpy as np
from multiagent.grid.grid_env import Env
from multiagent.grid.q_learning_agent import QLearningAgent
from multiagent.grid.random_action_agent import RandomActionAgent
from multiagent.grid.greedy_agent import GreedyAgent
from multiagent.grid.deep_reinforce_agent import DeepReinforceAgent
from multiagent.grid.central_controler import CentralController
import copy


EPISODES = 1000


if __name__ == "__main__":
    max_agent_count = 10
    max_victim_count = 10

    env = Env(max_agent_count, max_victim_count)

    agent_count = 1
    victim_count = 3

    # reproduce reinforcement
    agent = DeepReinforceAgent(actions=list(range(env.n_actions)), agent_id=0, env=env)
    env.add_agent_at_pos(agent, 0)

    env.add_victim_at_pos(5, -1)
    env.add_victim_at_pos(11, -1)
    env.add_victim_at_pos(17, -1)
    env.add_victim_at_pos(24, 1)

    # for i in range(agent_count):
    #     agent = DeepReinforceAgent(actions=list(range(env.n_actions)), agent_id=i, env=env)
    #     env.add_agent(agent)
    #
    # for i in range(victim_count):
    #     env.add_victim()

    env.pack_canvas()

    scheduler = CentralController(env, env.get_agents(), env.get_victims())

    for episode in range(EPISODES):
        state_n = env.reset_n()
        print("Episode", episode, "states:", state_n)
        counter = 0
        cumulative_reward = 0
        while True:
            env.render()
            done = False
            counter = counter + 1

            done_n = np.zeros(agent_count)
            reward_n = np.zeros(agent_count)
            action_n = scheduler.get_action(state_n)

            # take action and proceed one step in the environment
            for i in range(agent_count):
                agent = env.get_agent(i)
                # state = str(state_n[i])
                action = action_n[i]

                next_state, reward, done_i = env.agent_step(agent, action)
                done_n[i] = done_i
                state_n[i] = copy.deepcopy(next_state)
                reward_n[i] = reward

                cumulative_reward += reward
                print("Episode=", episode, ", agent=", agent.get_id(),  ", at iteration=", counter, ", with total reward=", cumulative_reward)

                # env.print_value_all(agent.q_table)

            scheduler.append_sample(state_n, action_n, reward_n)
            done = (np.sum(done_n) == len(done_n))
            # if episode ends, then break
            if done:

                scheduler.train_model()

                print("Episode=", episode, ", ends in a number of iterations=", counter, ", with total reward=", cumulative_reward)
                break
