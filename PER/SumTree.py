from Node import Node


class SumTree:
    def __init__(self, size):
        self.writer = 0
        self.size = size
        self.exp_nodes, self.base_node = self.construct_tree(size)
        self.n_filled_leaves = 0
        self.w_cursor = 0

    def _retrieve(self, value, node) -> Node:

        if node.left is None:
            return node

        if node.left.value >= value:
            return self._retrieve(value, node.left)

        else:
            return self._retrieve(value - node.left.value, node.right)

    def retrieve(self, value) -> Node:
        """

        :param value: value to search
        :return: Node
        """

        return self._retrieve(value, self.base_node)

    @staticmethod
    def construct_tree(size):
        """
        Construct tree from the amount of leaves 

        :param size: size of the nodes at the end of tree
        :return: tree and root node
        """

        nodes = [Node(None, None) for _ in range(size)]

        exp_nodes = nodes.copy()
        while len(nodes) > 1:
            nodes = [Node(nodes[idx1], nodes[idx2]) for idx1, idx2 in
                     zip(range(0, len(nodes), 2), range(1, len(nodes), 2))]

        return exp_nodes, nodes[0]

    def update_node(self, value, node, experience=None):
        change = value - node.value
        node.value = value
        if experience is not None:
            node.data = experience
        self.update_tree(node.parent, change)

    def update_tree(self, node, change):
        node.value = node.value + change

        if node.parent is not None:
            self.update_tree(node.parent, change)

    def append(self, value, experience):

        self.update_node(value, self.exp_nodes[self.w_cursor], experience)

        self.w_cursor += 1

        if self.w_cursor == self.size:
            self.w_cursor = 0

        if self.n_filled_leaves <= self.size:
            self.n_filled_leaves += 1

    def __len__(self):
        return self.n_filled_leaves
