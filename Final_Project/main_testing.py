from testing import Agent
import gym
import numpy as np
import os
# from utils import plotLearning

current_directory = os.getcwd()

env = gym.make('LunarLanderContinuous-v2')

agent = Agent(alpha=0.000025, beta=0.00025, input_dims=8, tau=0.001, env=env, batch_size=128, layer1_size=400, layer2_size=300, n_actions=2)

np.random.seed(0)

score_history = []
for i in range(2000):
    done = False
    score = 0
    obs = env.reset()
    obs = obs[0]

    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info, _ = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state

    score_history.append(score)
    print('episode ', i, ' score %.2f' % score, '100 game average %.2f' % np.mean(score_history[-100:]))

    if i % 25 == 0:
        agent.save_models()

filename = 'lunar-lander.png'
