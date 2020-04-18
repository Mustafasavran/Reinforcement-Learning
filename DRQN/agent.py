# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:13:04 2020

@author: mustafa
"""
from ExperienceBuffer import ExperienceReplay
from model import DRQN
import config as cf
import torch
import numpy as np
class Agent:
    def __init__(self,env,n_input,n_output):
        self.env=env
        self.epsilon=1.0
        self.epsilon_decay=0
        self.net=DRQN(n_input,n_output).to(cf.DEVICE)
        self.tgt_net=DRQN(n_input,n_output).to(cf.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cf.LEARNING_RATE)
        
    def action(self,state, hidden):
        
        state = state.unsqueeze(0).unsqueeze(0)
        q_value, hidden = self.tgt_net.forward(state, hidden)
        _, action = torch.max(q_value, 2)
        self.epsilon_decay+=1
        self.update_epsilon()
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample(), hidden
        else:
            return action.item(), hidden
        
        
    def update_epsilon(self):
        
        if self.epsilon_decay>1000:
            self.epsilon_decay=0
            self.epsilon=max(self.epsilon-1e-4,0.02)
            
            
            
    def update_tgt(self):
        
        self.tgt_net.load_state_dict(self.net.state_dict())
    
    def train_model(self, batch):
        current_states,rewards,actions,next_states,dones=batch
        

        
        states_v = torch.stack(current_states).view(cf.BATCH_SIZE, cf.l_sequence, self.net.n_input)
        next_states_v = torch.stack(next_states).view(cf.BATCH_SIZE, cf.l_sequence, self.net.n_input)
        actions_v = torch.stack(actions).view(cf.BATCH_SIZE, cf.l_sequence, -1).long()
        rewards_v = torch.stack(rewards).view(cf.BATCH_SIZE, cf.l_sequence, -1)
        dones_v=torch.stack(dones).view(cf.BATCH_SIZE, cf.l_sequence, -1)
        state_action_values,_ = self.net(states_v)
        
        state_action_values=state_action_values.gather(2, actions_v)
        next_state_values,_ = self.tgt_net(next_states_v)
        next_state_values=next_state_values.max(2, keepdim=True)[0]
        next_state_values = next_state_values.detach()
    
        expected_state_action_values = dones_v*cf.gamma*next_state_values  + rewards_v
        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
            
    
        
    
    
        
