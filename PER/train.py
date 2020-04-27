import gym
import numpy as np
from collections import deque
import torch
from PER import PER

import config as cf
from agent import Agent

buffer = PER(cf.BUFFER_SIZE)

env = gym.make(cf.ENV_NAME)
episode_rewards = deque(maxlen=10)

agent = Agent(env, 2, env.action_space.n, buffer=buffer)

agent.update_tgt()

print_flag = 1000

steps = 0

for epoch in range(10000):
    done = False

    episode_reward = 0
    state = env.reset()
    state = torch.Tensor(state).to(cf.DEVICE)

    while not done:
        steps += 1

        action = agent.action(state)
        next_state, reward, done, _ = env.step(action)

        next_state = torch.Tensor(next_state).to(cf.DEVICE)
        if done:
            mask = 1
        else:
            mask = 0

        episode_reward += reward
        agent.append_exp(state, action, next_state, reward, mask)
        state = next_state

        if steps > 1000 and len(buffer.tree) > cf.BATCH_SIZE:

            batch, is_weights, nodes = buffer.sample(cf.BATCH_SIZE)

            agent.train_model(batch, is_weights, nodes)

            if steps % 100 == 0:
                agent.update_tgt()

    episode_rewards.append(episode_reward)

    if epoch % 10 == 0:
        print("epoch: " + str(epoch) + " mean reward: " + str(np.mean(episode_rewards)) + " epsilon: " + str(
            agent.epsilon) + " reward: " + str(episode_reward))

    if np.mean(episode_rewards) > -100:
        break
