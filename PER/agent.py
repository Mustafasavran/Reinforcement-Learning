from model import DQN
import torch
import config as cf
import numpy as np
from collections import namedtuple

experience = namedtuple("Experience", ["state", "action", "next_state", "reward", "done"])


class Agent:
    def __init__(self, env, n_input, n_output, buffer):
        self.env = env
        self.epsilon = 1.0
        self.epsilon_decay = 0
        self.net = DQN(n_input, n_output).to(cf.DEVICE)
        self.tgt_net = DQN(n_input, n_output).to(cf.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cf.LEARNING_RATE)
        self.buffer = buffer

    def action(self, state):

        state = state.unsqueeze(0).unsqueeze(0)
        q_value = self.net.forward(state)
        _, action_ = torch.max(q_value, 2)

        self.epsilon_decay += 1
        self.update_epsilon()
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:

            return action_.item()

    def update_epsilon(self):

        if self.epsilon_decay > 1000:
            self.epsilon = max(self.epsilon - 0.00005, 0.02)

    def update_tgt(self):

        self.tgt_net.load_state_dict(self.net.state_dict())

    def train_model(self, batch, is_weights, nodes):
        states_v, actions_v, next_states_v, rewards_v, dones_v = batch

        states_v = torch.tensor(states_v)
        next_states_v = torch.tensor(next_states_v)
        actions_v = torch.tensor(actions_v).long()
        rewards_v = torch.tensor(rewards_v)
        dones_v = torch.ByteTensor(dones_v)

        states_v = states_v.view(cf.BATCH_SIZE, self.net.n_input)
        next_states_v = next_states_v.view(cf.BATCH_SIZE, self.net.n_input)
        rewards_v = rewards_v.view(cf.BATCH_SIZE, -1).to(cf.DEVICE)
        dones_v = dones_v.view(cf.BATCH_SIZE, -1).to(cf.DEVICE)
        state_action_values = self.net(states_v)

        state_action_values = state_action_values.gather(1, actions_v.to(cf.DEVICE).unsqueeze(-1)).squeeze(-1)
        next_state_values = self.tgt_net(next_states_v)

        next_state_values = next_state_values.max(1, keepdim=True)[0]

        next_state_values = next_state_values.detach()

        expected_state_action_values = (1 - dones_v) * cf.gamma * next_state_values + rewards_v
        errors = torch.abs(state_action_values - expected_state_action_values)
        for i in range(cf.BATCH_SIZE):
            self.buffer.update(errors[0][i].item(), nodes[i])
        self.optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(state_action_values.double().view(cf.BATCH_SIZE, 1),
                                            expected_state_action_values.double())

        loss = (torch.DoubleTensor(is_weights).to(cf.DEVICE) * loss).mean()
        loss.backward()
        self.optimizer.step()

        return loss

    def append_exp(self, state, action, next_state, reward, done):
        target = self.net(state)
        old_val = torch.max(target)
        target_val = self.tgt_net(next_state)
        if not done:
            expected_reward = reward
        else:
            expected_reward = reward + cf.gamma * torch.max(target_val)

        error = abs(old_val - expected_reward)

        self.buffer.append(error.item(), experience(state, action, next_state, reward, done))
