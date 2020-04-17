# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 07:17:47 2020

@author: mustafa
"""


import torch

ENV_NAME = 'CartPole-v1'
gamma = 0.99
BATCH_SIZE = 32
LEARNING_RATE= 0.001
BUFFER_SIZE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

l_sequence= 8
