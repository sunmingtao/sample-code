import numpy as np
import math

text_file = open("projecteuler/p107.txt", "r")
lines = text_file.read().split('\n')
text_file.close()


matrix = []
for line in lines:
    matrix.append([int(cell) if cell != '-' else 0 for cell in line.split(',')])

matrix = np.array(matrix)

test_matrix = np.array([[0, 16, 12, 21, 0, 0, 0],
                       [16, 0, 0, 17, 20, 0, 0],
                       [12, 0, 0, 28, 0, 31, 0],
                       [21, 17, 28, 0, 18, 19, 23],
                       [0, 20, 0, 18, 0, 0, 11],
                       [0, 0, 31, 19, 0, 0, 27],
                       [0, 0, 0, 23, 11, 27, 0]])


def find_smallest_edge(matrix):
    smallest_i = 0
    smallest_j = 0
    smallest_edge = math.inf
    for i in range(len(matrix)):
        for j in range(i):
            if matrix[i,j] != 0 and matrix[i,j] < smallest_edge:
                smallest_edge = matrix[i, j]
                smallest_i = i
                smallest_j = j
    return smallest_i, smallest_j, smallest_edge

assert find_smallest_edge(test_matrix) == (6, 4, 11)



def find_smallest_edge_from_nodes(matrix, nodes):
    smallest_i = 0
    smallest_j = 0
    smallest_edge = math.inf
    for node in nodes:
        for other_node in range(len(matrix)):
            if other_node not in nodes and matrix[node, other_node] != 0 and matrix[node, other_node] < smallest_edge:
                smallest_i = node
                smallest_j = other_node
                smallest_edge = matrix[node, other_node]
    return smallest_i, smallest_j, smallest_edge


assert find_smallest_edge_from_nodes(test_matrix, (6, 4)) == (4, 3, 18)
assert find_smallest_edge_from_nodes(test_matrix, (6, 4, 3)) == (3, 1, 17)

def find_smallest_edges(matrix):
    matrix_len = len(matrix)
    node1, node2, sum_edge = find_smallest_edge(matrix)
    nodes = [node1, node2]
    while len(nodes) < matrix_len:
        _, new_node, edge = find_smallest_edge_from_nodes(matrix, nodes)
        sum_edge += edge
        nodes.append(new_node)
    return sum_edge

assert find_smallest_edges(test_matrix) == 93


def find_max_saving_edges(matrix):
    return np.sum(matrix) // 2 - find_smallest_edges(matrix)

assert find_max_saving_edges(test_matrix) == 150


print(find_max_saving_edges(matrix))

