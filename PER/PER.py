from SumTree import SumTree
import numpy as np


class PER(object):
    def __init__(self, size, min_prob=1e-4, alpha=0.8, beta=0.8, beta_inc=1e-5):
        self.tree = SumTree(size)
        self.min_prob = min_prob
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc

    def calculate_priority(self, error) -> float:
        return pow(abs(error) + self.min_prob, self.alpha)

    def update_beta(self):
        self.beta = self.beta + self.beta_inc

    def append(self, value, experience):
        prior = self.calculate_priority(value)
        self.tree.append(prior, experience)

    def sample(self, batch_size=32):
        states = []
        next_states = []
        rewards = []
        dones = []
        actions = []
        values = []

        sample_values = np.random.uniform(self.tree.base_node.value, size=batch_size)
        nodes = []

        for val in sample_values:
            node = self.tree.retrieve(val)

            states.append(node.data.state.tolist())
            next_states.append(node.data.next_state.tolist())
            rewards.append(node.data.reward)
            dones.append(node.data.done)
            actions.append(node.data.action)
            values.append(node.value)
            nodes.append(node)

        samp_prob = np.array(values) / self.tree.base_node.value
        is_weights = pow(1 / (self.tree.n_filled_leaves * samp_prob), self.beta)
        is_weights = is_weights / max(is_weights)

        self.update_beta()

        return (np.array(states), np.array(actions), np.array(next_states), np.array(
            rewards), np.array(dones)), is_weights, nodes

    def update(self, value, node):

        self.tree.update_node(self.calculate_priority(value), node)

