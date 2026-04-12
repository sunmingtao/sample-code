import math
import numpy as np

class Node:
    visited=False
    weight=math.inf
    x, y = 0, 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_visited(self):
        self.visited = True

    def set_weight(self, new_weight):
        self.weight = new_weight

    def is_visited(self):
        return self.visited

    def get_weight(self):
        return self.weight

    def __repr__(self):
        return "Node(%s, %s)" % (self.x, self.y)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


def init_nodes():
    all_nodes = []
    for i in range(size):
        node_row = []
        for j in range(size):
            node_row.append(Node(i, j))
        all_nodes.append(node_row)
    return all_nodes


def init_matrix():
    text_file = open("p82.txt", "r")
    lines = text_file.read().replace('\n', ',').split(',')
    text_file.close()
    return np.array(lines).reshape(size, size).astype(int).tolist()


def update_shortest_path(node):
    x = node.x
    y = node.y
    update_neighbour_shortest_path(node, x-1, y)
    update_neighbour_shortest_path(node, x+1, y)
    update_neighbour_shortest_path(node, x, y-1)
    update_neighbour_shortest_path(node, x, y+1)
    node.set_visited()
    active_nodes.remove(node)

def update_neighbour_shortest_path(node, n_x, n_y):
    if 0 <= n_x < size and 0 <= n_y < size:
        n_node = all_nodes[n_x][n_y]
        if not n_node.is_visited():
            active_nodes.add(n_node)
            new_weight = node.get_weight() + arr[n_node.x][n_node.y]
            if new_weight < n_node.get_weight():
                print(n_node, 'weight updated to', new_weight)
                n_node.set_weight(new_weight)



def get_next_active_node():
    shortest = math.inf
    for n in active_nodes:
        if n.get_weight() < shortest:
            shortest = n.get_weight()
            next_node_ = n
    print('next node is ', next_node_)
    return next_node_



size = 80
visited_nodes = set()
active_nodes = set()
all_nodes = init_nodes()
arr = init_matrix()
start = all_nodes[0][0]
end = all_nodes[size-1][size-1]

start.set_weight(arr[start.x][start.y])
active_nodes.add(start)

update_shortest_path(start)
while not end.is_visited():
    next_node = get_next_active_node()
    update_shortest_path(next_node)
print('Shortest path is',end.get_weight())



