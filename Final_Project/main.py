import DDPG
import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("LunarLanderContinuous-v2")
# env = gym.make("Pendulum-v1")

# env = gym.wrappers.RecordEpisodeStatistics(env)

num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
# action_bound = env.action_space.high
gamma = 0.95
tau = 1e-2
num_nodes = 256
critic_alpha = 1e-3
actor_alpha = 1e-4
batch_size = 128

num_episodes = 50

DDPG_Agent = DDPG.Agent(num_state, num_action, gamma, tau, num_nodes, critic_alpha, actor_alpha, batch_size)
OU_noise = DDPG.OUNoise(env.action_space)

rewards = []
avg_rewards = []

for ep in range(num_episodes):
    state = env.reset()
    state = state[0]
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    OU_noise.reset()
    ep_reward = 0

    for t in itertools.count():
        action = DDPG_Agent.choose_action(state)
        action = OU_noise.get_action(action, t)
        next_state, reward, done, _ , _ = env.step(action)
        action = torch.tensor(action, dtype=torch.float, device=device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float, device=device).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float, device=device).unsqueeze(0)
        # print(state[0], next_state)
        DDPG_Agent.memory.add(state, action, next_state, reward)

        state = next_state
        ep_reward += reward

        DDPG_Agent.optimize()

        if done:
            # sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(ep_reward)
    # avg_rewards.append(np.mean(rewards[-10:]))
