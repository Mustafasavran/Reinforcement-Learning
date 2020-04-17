# -*- coding: utf-8 -*-

    
  
import gym
import numpy as np
from collections import deque
import torch
from ExperienceBuffer import ExperienceReplay

import config as cf
from agent import Agent



env = gym.make(cf.ENV_NAME)
episode_rewards=deque(maxlen=100)


agent=Agent(env,3,env.action_space.n)


agent.update_tgt()

print_flag=1000

steps = 0
buffer = ExperienceReplay(cf.BUFFER_SIZE)

for epoch in range(10000):
    done = False
    
    episode_reward = 0
    state = env.reset()
    state = state[0:3]
    state = torch.Tensor(state).to(cf.DEVICE)

    hidden = None

    while not done:
        steps += 1

        action, hidden = agent.action(state, hidden)
        next_state, reward, done, _ = env.step(action)

        next_state = next_state[0:3]
        next_state = torch.Tensor(next_state).to(cf.DEVICE)
        if done:
            mask = 0
        else:
            mask = 1
        
        
        episode_reward += reward
        buffer.add(state,reward,action,next_state,mask)
        state = next_state

        
        if steps >1000 and len(buffer.buffer) > cf.BATCH_SIZE:
                

            batch = buffer.sample(cf.BATCH_SIZE)
            agent.train_model( batch)

            if steps % 100 == 0:
                agent.update_tgt()

    episode_rewards.append(episode_reward)
    
    if epoch % 10 == 0:
        print("epoch: "+str(epoch)+" mean reward: "+str(np.mean(episode_rewards))+" epsilon: "+str(agent.epsilon)+" reward: "+str(episode_reward))
        
    if np.mean(episode_reward) > 195:
        break

