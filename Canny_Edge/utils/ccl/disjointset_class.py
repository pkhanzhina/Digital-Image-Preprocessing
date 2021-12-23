from Canny_Edge.utils.ccl.node_class import Node


class DisjointSet:
    def __init__(self):
        self.nodes = {}

    def MakeSet(self, value):
        exist_node = self.nodes.get(value)
        if exist_node is None:
            exist_node = Node(value)
            self.nodes[value] = exist_node
        return exist_node

    def Find(self, value):
        x = self.nodes.get(value)
        root = x
        while root.parent != root:
            root = root.parent
        while x.parent != root:
            parent = x.parent
            x.parent = root
            x = parent
        return root

    def Union(self, value_r, value_s):
        if value_r == value_s:
            return
        root_r = self.Find(value_r)
        root_s = self.Find(value_s)
        if root_r != root_s:
            if root_r.rank < root_s.rank:
                root_r.parent = root_s
            else:
                root_s.parent = root_r
                if root_r.rank == root_s.rank:
                    root_s.rank += 1
