# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:19:15 2020

@author: mustafa
"""

  
import torch.nn as nn
import config as cf
class DRQN(nn.Module):
    def __init__(self,n_input,n_output):
        super(DRQN, self).__init__()
        self.n_input=n_input
        self.n_output=n_output
        self.lstm = nn.LSTM(input_size=n_input, hidden_size=128, batch_first=True)
        
        self.network=nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,n_output)
            )
    
    def forward(self,x,h=None):
        x=x.to(cf.DEVICE)
        if h==None:
            o, h = self.lstm(x)
        
        else:
            o,h=self.lstm(x,h)
        
        return self.network(o),h
    
    
    

        
        
        
            
            
        
        
