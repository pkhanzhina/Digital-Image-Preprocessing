import numpy as np
from Canny_Edge.utils.ccl.disjointset_class import DisjointSet


class CCL:
    def __init__(self, con_type=8):
        if con_type == 4:
            self.neighbours = [[0, -1], [-1, 0]]
        elif con_type == 8:
            self.neighbours = [[0, -1], [-1, -1], [-1, 0], [-1, 1]]
        else:
            raise Exception("Unknown connection type")

    def __call__(self, img):
        disjoint_set = DisjointSet()
        labels = self.first_pass(img, disjoint_set)
        final_labels = self.second_pass(labels, disjoint_set)
        return final_labels

    def first_pass(self, img, disjoint_set):
        img = np.pad(img, ((1, 1), (1, 1)), constant_values=[0, 0])
        labels = np.zeros_like(img, dtype=np.int)
        h, w = img.shape
        cur_label = 1
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                if img[x, y] == 0:
                    continue
                nn_labels = self._get_neighboring_labels(img[x - 1:x + 1, y - 1:y + 2],
                                                         labels[x - 1:x + 1, y - 1:y + 2])
                if len(nn_labels) == 0:
                    disjoint_set.MakeSet(cur_label)
                    labels[x, y] = cur_label
                    cur_label += 1
                else:
                    min_label = min(nn_labels)
                    labels[x, y] = min_label
                    for label in nn_labels:
                        disjoint_set.Union(min_label, label)
        return labels[1:-1, 1:-1]

    def second_pass(self, labels, disjoint_set):
        h, w = labels.shape
        for x in range(h):
            for y in range(w):
                if labels[x, y] == 0:
                    continue
                labels[x, y] = disjoint_set.Find(labels[x, y]).value
        return labels

    def _get_neighboring_labels(self, nn, labels):
        nn_labels = []
        x, y = 1, 1
        for shift in self.neighbours:
            nx, ny = x + shift[0], y + shift[1]
            if nn[nx, ny] == 0:
                continue
            nn_labels.append(labels[nx, ny])
        return nn_labels

