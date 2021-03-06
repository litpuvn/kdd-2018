#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

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
import pylab
from multiagent.grid.distribution import Distribution
import logging


# if __name__ == '__main__':
#
#
#     # load scenario from script
#     scenario = SpeakerListenerScenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
#     # render call to create viewer window (necessary only for interactive policies)
#     env.render()
#     # create interactive policies for each agent
#     policies = [InteractivePolicy(env,i) for i in range(env.n)]
#     # execution loop
#     obs_n = env.reset()
#     while True:
#         # query for action from each agent's policy
#         act_n = []
#         for i, policy in enumerate(policies):
#             act_n.append(policy.action(obs_n[i]))
#         # step environment
#         obs_n, reward_n, done_n, _ = env.step(act_n)
#         # render all agent views
#         env.render()
#         # display rewards
#         #for agent in env.world.agents:
#         #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))


TOTAL_EPISODES = 12500
POLICY_RANDOM = 'random'
POLICY_GREEDY = 'greedy'

if __name__ == "__main__":
    max_agent_count = 50
    max_victim_count = 50

    env = Env(max_agent_count, max_victim_count)

    # random, greedy
    TEST_POLICY = POLICY_RANDOM

    agent_count = 2
    victim_count = 2
    logger = Env.setup_custom_logger("app", logging.INFO, TEST_POLICY + "_policy_log.txt")

    distribution = Distribution()

    volunteer_distribution = distribution.get_distribution_of_volunteers()
    agent_count = len(volunteer_distribution)

    for i in range(agent_count):

        if TEST_POLICY == POLICY_RANDOM:
            agent = RandomActionAgent(actions=list(range(env.n_actions)), agent_id=i, env=env, options={'distributed': False})
        elif TEST_POLICY == POLICY_GREEDY:
            agent = GreedyAgent(actions=list(range(env.n_actions)), agent_id=i, env=env, options={'distributed': False})
        else:
            raise Exception('Invalid policy to run')

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

    global_step = 0
    episodes = []
    scores = []
    episode_time_steps = []

    # logFile = open('./save_graph/log.txt', 'a')

    for episode in range(TOTAL_EPISODES):
        state_n = env.reset_n()
        counter = 0
        cumulative_reward = 0

        score = 0
        episode_time_step = 0

        while True:
            env.render()
            global_step += 1
            episode_time_step += 1
            score_per_episode_time_step = 0

            done = False
            reward_n = np.zeros(agent_count)
            counter = counter + 1
            # take action and proceed one step in the environment
            for i in range(agent_count):
                agent = env.get_agent(i)
                state = str(state_n[i])
                action = agent.get_action(state)
                next_state, reward, done = env.agent_step(agent, action)

                # with sample <s,a,r,s'>, agent learns new q function
                agent.learn(str(state), action, reward, str(next_state))

                state_n[i] = next_state
                reward_n[i] = reward

                cumulative_reward += reward

                score += reward
                score_per_episode_time_step += reward

            # print("episode:", episode, " episode time_step:", episode_time_step, " score:", score_per_episode_time_step)

            # if episode ends, then break
            if done:
                scores.append(score)
                episode_time_steps.append(episode_time_step)
                episodes.append(episode)
                logger.info("episode:" + str(episode) + "  score:" + str(score) + "  episode time_step:" + str(episode_time_step) + " global time:" + str(global_step))

                break

        if episode % 10 == 0:
            pylab.figure(1)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/" + TEST_POLICY + "_policy_score.png")

            pylab.figure(2)
            pylab.plot(episodes, episode_time_steps, 'b')
            pylab.savefig("./save_graph/" + TEST_POLICY + "_policy_time_step.png")
