class NeighboringPoints:
    def __init__(self):
        self.points = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        self.n = len(self.points)
        self.nrof_point = {self.points[i]: i for i in range(0, self.n)}

    def get(self, start_with, clockwise=True):
        start_with = (start_with[0], start_with[1])
        nr = self.nrof_point[start_with]
        dir = 1 if clockwise else -1
        for i in range(self.n):
            next_p = (nr + dir + self.n) % self.n
            nr = next_p
            yield self.points[next_p]

