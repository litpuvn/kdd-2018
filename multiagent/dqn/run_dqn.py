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
from multiagent.dqn.grid_env import Env
from multiagent.dqn.dqn_agent import DQNAgent
from multiagent.dqn.dqn_policy import DQNPolicy
from multiagent.dqn.memory import Memory

import pylab
from multiagent.grid.distribution import Distribution
import logging

TOTAL_EPISODES = 15000

if __name__ == "__main__":
    max_agent_count = 50
    max_victim_count = 50

    info = {
        "env": {"Ny": 4,
                "Nx": 4},
        "agent": {"policy_mode": "epsgreedy",  # "epsgreedy", "softmax"
                  "eps": 1.0,
                  "eps_decay": 2.0 * np.log(10.0) / TOTAL_EPISODES},
        "brain": {"discount": 0.99,
                  "learning_rate": 0.9},
        "memory": {}
    }

    env = Env(max_agent_count, max_victim_count, info)

    agent_count = 2
    victim_count = 2

    distribution = Distribution()

    volunteer_distribution = distribution.get_distribution_of_volunteers()
    agent_count = len(volunteer_distribution)

    for i in range(agent_count):
        agent = DQNAgent(actions=list(range(env.n_actions)), agent_id=i, env=env, options={'distributed': False})
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

    memory = Memory(info)
    policy = DQNPolicy(env, info)
    logger = Env.setup_custom_logger("app", logging.INFO)
    q_table_logger = Env.setup_custom_logger("qtable", logging.INFO, 'q_table.log')
    global_step = 0
    episodes = []
    scores = []
    episode_time_steps = []

    for episode in range(TOTAL_EPISODES):
        # state_n is position of each agent {agent_0: [r1, c1], agent_1: [r2, c2]}
        env.reset_n()
        counter = 0
        cumulative_reward = 0
        score = 0

        episode_time_step = 0

        while True:
            env.render()
            done = False
            # reward_n = np.zeros(agent_count)
            counter = counter + 1
            # take action and proceed one step in the environment
            global_step += 1
            episode_time_step += 1
            # next_state_n = copy.deepcopy(state_n)
            state_n = env.current_state()
            action_n = policy.get_action_n(state_n, episode=episode)
            next_state_n = []
            reward_n = []
            done_n = []

            for i in range(agent_count):
                agent = env.get_agent(i)
                action = action_n[i]
                state = state_n[i]
                next_state, reward, done = env.agent_step(agent, action)

                next_state_n.append(next_state)
                reward_n.append(reward)
                done_n.append(done)

                if state[0] != next_state[0] or state[1] != next_state[1]:
                    agent.set_last_action(action)

                cumulative_reward += reward
                score += reward

                # action_n.append(action)

            policy.learn(state_n, action_n, reward_n, next_state_n)

            if env.is_terminated() or sum(done_n) > 0:
                done = True

            # logger.info("state=" + str(state_n) + "; action=" + str(action_n) + "; reward=" + str(
            #     sum(reward_n)) + "; next_state=" + str(next_state_n))

            # state_n = copy.deepcopy(next_state_n)
            env.print_value_all(DQNPolicy.Q_TABLE)

            # if episode ends, then break
            if done:
                scores.append(score)
                episodes.append(episode)
                episode_time_steps.append(episode_time_step)

                # print("episode:", episode, "  score:", score, "  episode time_step:", episode_time_step, " global time:", global_step)

                logger.info("episode:" + str(episode) + "  score:" + str(score) + "  episode time_step:" + str(episode_time_step) + " global time:" + str(global_step))
                break

        # Update model when episode finishes
        # policy.update(memory, env)
        if episode % 10 == 0:
            pylab.figure(1)
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/q_policy_score.png")

            pylab.figure(2)
            pylab.plot(episodes, episode_time_steps, 'b')
            pylab.savefig("./save_graph/q_policy_time_step.png")

            # for log_r in policy.get_qtable():
            #     q_table_logger.info(log_r)

        # Clear memory for next episode
        # memory.clear_memory()