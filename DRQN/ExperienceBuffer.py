# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:35:52 2020

@author: mustafa
"""
import torch
from collections import namedtuple, deque
import numpy as np
from config import l_sequence
Exp=namedtuple("Experience", ["current_state","reward","action","next_state","done"])

class ExperienceReplay:
    
    def __init__(self,size=100):
        self.episode_buffer=[]
        self.size=size
        self.buffer=deque(maxlen=self.size)
        
    def add(self,current_state,reward,action,next_state,done):
        experience=Exp(current_state,reward,action,next_state,done)
        self.episode_buffer.append(experience)
        
        
        if  done==0:
            while len(self.episode_buffer) < l_sequence:# If Episode is not finish
                    self.episode_buffer.insert(0, Exp(torch.zeros_like(current_state),0,0,torch.zeros_like(next_state),0))
        
            self.buffer.append(self.episode_buffer)
            self.episode_buffer=[]
    
    def sample(self,size):
        
        states=[]
        next_states=[]
        rewards=[]
        dones=[]
        actions=[]
        indices = np.random.choice(len(self.buffer), size, replace=False)
        for index in indices:
            episode=self.buffer[index]
            if len(episode)!=l_sequence:
              start_episode=np.random.choice(range(len(episode)-l_sequence),1,replace=False)[0]
            else:
              start_episode=0
            
            experiences=episode[start_episode:start_episode+l_sequence]
            
            ep_states, ep_rewards,ep_actions, ep_next_states, ep_dones  = zip(*experiences)
            states.append(torch.stack(list(ep_states)))
            next_states.append(torch.stack(list(ep_next_states)))
            
            rewards.append(torch.Tensor(list(ep_rewards)))
            dones.append(torch.Tensor(list(ep_dones)))
            actions.append(torch.Tensor(list(ep_actions)))
        
        
        return states, rewards,actions,next_states,dones
    
    
                
                
                
                
                
            
            
            
        
        
            
