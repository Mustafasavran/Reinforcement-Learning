import torch
import torch.nn as nn
import config as cf


class DQN(nn.Module):
    def __init__(self, n_input, n_action):
        super(DQN, self).__init__()
        self.n_input = n_input
        self.n_output = n_action
        self.fc = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_action)
        )

    def forward(self, x):
        x = x.to(cf.DEVICE)
        return self.fc(x.float())
