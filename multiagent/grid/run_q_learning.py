#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios
from multiagent.speaker_listener import SpeakerListenerScenario
import copy
import numpy as np
from multiagent.grid.grid_env import Env
from multiagent.grid.q_learning_agent import QLearningAgent
from multiagent.grid.q_learning_policy import QLearningPolicy
from multiagent.grid.random_action_agent import RandomActionAgent
from multiagent.grid.greedy_agent import GreedyAgent
from multiagent.grid.deep_reinforce_agent import DeepReinforceAgent
import pylab
from multiagent.grid.distribution import Distribution
import logging

TOTAL_EPISODES = 102500

if __name__ == "__main__":
    max_agent_count = 50
    max_victim_count = 50

    env = Env(max_agent_count, max_victim_count)

    agent_count = 2
    victim_count = 4

    # # for i in range(agent_count):
    # #     agent = QLearningAgent(actions=list(range(env.n_actions)), agent_id=i, env=env)
    # #     env.add_agent(agent)
    # agent = QLearningAgent(actions=list(range(env.n_actions)), agent_id=0, env=env)
    # env.add_agent_at_pos(agent, 22)
    #
    # agent2 = QLearningAgent(actions=list(range(env.n_actions)), agent_id=1, env=env)
    # env.add_agent_at_pos(agent2, 22)
    #
    # # for i in range(victim_count):
    # #     env.add_victim()
    #
    # # env.add_victim_at_pos(7, -100)
    # # env.add_victim_at_pos(11, -100)
    # # env.add_victim_at_pos(12, 100)
    # # env.add_victim_at_pos(24, 100)
    #
    # # good for beating greedy first
    # env.add_victim_at_pos(5, 100)
    # env.add_victim_at_pos(4, 100)

    distribution = Distribution()

    volunteer_distribution = distribution.get_distribution_of_volunteers()
    agent_count = len(volunteer_distribution)

    for i in range(agent_count):
        agent = QLearningAgent(actions=list(range(env.n_actions)), agent_id=i, env=env, options={'distributed': False})
        row_col = volunteer_distribution[i]
        if len(row_col) != 2:
            raise Exception('Invalid volunteer position')

        env.add_agent_at_row_col(agent, row_col[0], row_col[1])

    victim_distribution = distribution.get_distribution_of_vitims()
    victim_count = len(victim_distribution)
    for i in range(victim_count):
        row_col = victim_distribution[i]
        if len(row_col) != 3:
            raise Exception('Invalid victim position')

        env.add_victim_at_row_col(row_col[0], row_col[1], row_col[2])

    env.pack_canvas()

    policy = QLearningPolicy(env)
    logger = Env.setup_custom_logger("app", logging.INFO)
    q_table_logger = Env.setup_custom_logger("qtable", logging.INFO, 'q_table.log')
    global_step = 0
    episodes = []
    scores = []
    episode_time_steps = []

    for episode in range(TOTAL_EPISODES):
        # state_n is position of each agent {agent_0: [r1, c1], agent_1: [r2, c2]}
        state_n = env.reset_n()
        counter = 0
        cumulative_reward = 0
        score = 0

        episode_time_step = 0

        while True:
            env.render()
            done = False
            reward_n = np.zeros(agent_count)
            counter = counter + 1
            # take action and proceed one step in the environment
            global_step += 1
            episode_time_step += 1
            next_state_n = copy.deepcopy(state_n)

            action_n = []
            action_n = policy.get_action_n(state_n)

            for i in range(agent_count):
                agent = env.get_agent(i)
                action = action_n[i]
                # action = policy.get_agent_action(i, next_state_n)
                state =  state_n[i]
                next_state, reward, done = env.agent_step(agent, action)

                if state[0] != next_state[0] or state[1] != next_state[1]:
                    agent.set_last_action(action)

                next_state_n[i] = copy.deepcopy(next_state)
                reward_n[i] = reward

                cumulative_reward += reward
                score += reward

                # action_n.append(action)

            policy.learn(state_n, action_n, reward_n, next_state_n)

            # logger.info("state=" + str(state_n) + "; action=" + str(action_n) + "; reward=" + str(
            #     reward_n) + "; next_state=" + str(next_state_n))


            state_n = copy.deepcopy(next_state_n)
            env.print_value_all(QLearningPolicy.Q_TABLE)

            # if episode ends, then break
            if done:
                scores.append(score)
                episodes.append(episode)
                episode_time_steps.append(episode_time_step)

                # print("episode:", episode, "  score:", score, "  episode time_step:", episode_time_step, " global time:", global_step)

                logger.info("episode:" + str(episode) + "  score:" + str(score) + "  episode time_step:" + str(episode_time_step) + " global time:" + str(global_step))
                break
        if episode % 10 == 0:
            pylab.figure(1)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/q_policy_score.png")

            pylab.figure(2)
            pylab.plot(episodes, episode_time_steps, 'b')
            pylab.savefig("./save_graph/q_policy_time_step.png")

            for log_r in policy.get_qtable():
                q_table_logger.info(log_r)