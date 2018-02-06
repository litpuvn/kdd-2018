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

TOTAL_EPISODES = 1000

if __name__ == "__main__":
    max_agent_count = 10
    max_victim_count = 10

    env = Env(max_agent_count, max_victim_count)

    agent_count = 2
    victim_count = 4

    # for i in range(agent_count):
    #     agent = QLearningAgent(actions=list(range(env.n_actions)), agent_id=i, env=env)
    #     env.add_agent(agent)
    agent = QLearningAgent(actions=list(range(env.n_actions)), agent_id=0, env=env)
    env.add_agent_at_pos(agent, 0)

    agent2 = QLearningAgent(actions=list(range(env.n_actions)), agent_id=1, env=env)
    env.add_agent_at_pos(agent2, 20)

    # for i in range(victim_count):
    #     env.add_victim()
    env.add_victim_at_pos(7, 100)
    env.add_victim_at_pos(11, 100)
    env.add_victim_at_pos(12, 100)
    env.add_victim_at_pos(24, 100)

    env.pack_canvas()

    policy = QLearningPolicy(env, list(range(env.n_actions)))

    global_step = 0
    episodes = []
    scores = []
    episode_time_steps = []

    for episode in range(TOTAL_EPISODES):
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

            policy.learn(state_n, action_n, reward_n, next_state_n)
            state_n = copy.deepcopy(next_state_n)

            # if episode ends, then break
            if done:
                scores.append(score)
                episodes.append(episode)
                episode_time_steps.append(episode_time_step)

                print("episode:", episode, "  score:", score, "  episode time_step:", episode_time_step, " global time:", global_step)
                break
        if episode % 10 == 0:
            pylab.figure(1)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/q_policy_score.png")

            pylab.figure(2)
            pylab.plot(episodes, episode_time_steps, 'b')
            pylab.savefig("./save_graph/q_policy_time_step.png")