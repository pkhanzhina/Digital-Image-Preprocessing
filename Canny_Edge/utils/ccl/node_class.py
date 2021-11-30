class Node:
    def __init__(self, value):
        self.value = value
        self.parent = self
        self.rank = 0

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return f"value: {self.value}, parent's value: {self.parent.value}, rank: {self.rank}"
