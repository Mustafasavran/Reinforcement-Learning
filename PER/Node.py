

class Node(object):
    def __init__(self, left, right, data=None):
        self.left = left
        self.right = right
        self.parent = None
        self.value = self.update_value(self.left, self.right)
        self.make_relation(self.left, self.right)
        self.data = data

    def update_value(self, left, right) -> float:
        value1, value2 = 0., 0.
        if left is not None:
            value1 = left.value
        if right is not None:
            value2 = right.value
        return value1 + value2

    def make_relation(self, left, right):
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self
