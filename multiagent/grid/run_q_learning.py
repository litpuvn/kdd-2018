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

TOTAL_EPISODES = 2500

if __name__ == "__main__":
    max_agent_count = 10
    max_victim_count = 10

    env = Env(max_agent_count, max_victim_count)

    agent_count = 2
    victim_count = 3

    for i in range(agent_count):
        agent = QLearningAgent(actions=list(range(env.n_actions)), agent_id=i, env=env)
        env.add_agent(agent)

    for i in range(victim_count):
        env.add_victim()

    env.pack_canvas()

    policy = QLearningPolicy(env, list(range(env.n_actions)))

    global_step = 0
    episodes = []
    scores = []

    for episode in range(TOTAL_EPISODES):
        state_n = env.reset_n()
        print("Episode", episode, "states:", state_n)
        counter = 0
        cumulative_reward = 0
        score = 0

        while True:
            env.render()
            done = False
            reward_n = np.zeros(agent_count)
            counter = counter + 1
            # take action and proceed one step in the environment
            global_step += 1
            action_n = policy.get_action_n(state_n)
            next_state_n = []

            for i in range(agent_count):
                agent = env.get_agent(i)
                state = str(state_n[i])
                action = action_n[i]
                next_state, reward, done = env.agent_step(agent, action)
                next_state_n.append(next_state)
                reward_n[i] = reward

                cumulative_reward += reward
                score += reward
                # print("Episode=", episode, ", agent=", agent.get_id(),  ", at iteration=", counter, ", with total reward=", cumulative_reward)

                # env.print_value_all(agent.q_table)

            policy.learn(state_n, action_n, reward_n, next_state_n)
            state_n = copy.deepcopy(next_state_n)

            # if episode ends, then break
            if done:
                scores.append(score)
                episodes.append(episode)

                print("episode:", episode, "  score:", score, "  time_step:", global_step)

                # print("Episode=", episode, ", ends in a number of iterations=", counter, ", with total reward=", cumulative_reward)
                break
        if episode % 10 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/q_policy.png")
