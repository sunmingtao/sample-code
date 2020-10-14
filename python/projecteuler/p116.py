import numpy as np

def init_matrix(block_size, max_unit):
    max_row = max_unit // block_size
    matrix = np.zeros((max_row + 1, max_unit + 1), dtype=np.long)
    for row in range(1, max_row + 1):
        for col in range(2, max_unit + 1):
            if row == 1:
                matrix[row, col] = col - block_size + 1
            else:
                room = col - row * block_size
                if room >= 0:
                    matrix[row, col] = sum(matrix[row - 1, block_size * (row - 1) + i] for i in range(0, room + 1))
    return matrix


matrix = init_matrix(2, 5)

assert matrix[1,2] == 1
assert matrix[1,3] == 2
assert matrix[1,4] == 3
assert matrix[1,5] == 4
assert matrix[2,4] == 1
assert matrix[2,5] == 3

matrix = init_matrix(2, 50)
assert matrix[1,50] == 49
assert matrix[2,7] == 10
assert matrix[3,6] == 1
assert matrix[3,7] == 4
assert matrix[4,8] == 1

matrix = init_matrix(3, 50)
assert matrix[1, 9] == 7
assert matrix[2, 5] == 0
assert matrix[2, 6] == 1
assert matrix[2, 9] == 10
assert matrix[3, 9] == 1

def ways(block_size, max_unit):
    matrix = init_matrix(block_size, max_unit)
    return np.sum(matrix, axis=0)[max_unit]

assert ways(2, 5) == 7
assert ways(3, 5) == 3
assert ways(4, 5) == 2

print (ways(2,50)+ways(3,50)+ways(4,50))