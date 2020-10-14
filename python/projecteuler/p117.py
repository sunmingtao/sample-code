import numpy as np

def sum_row_up_to_col(matrix, row, col):
    return sum(matrix[row, i] for i in range(0, col+1))


matrix = np.ones((4, 4), dtype=np.long)
assert sum_row_up_to_col(matrix, 2, 3) == 4


def init_matrix(max_unit):
    max_row = max_unit // 2
    matrix = np.zeros((max_row + 1, max_unit + 1), dtype=np.long)
    for row in range(1, max_row + 1):
        for col in range(2, max_unit + 1):
            if row == 1:
                if col == 2:
                    matrix[row, col] = 1
                else:
                    matrix[row, col] = (col - 2) * 3
            else:
                if col >= 4:
                    matrix[row, col] = sum(sum_row_up_to_col(matrix, row-1, i) for i in range(col - 2, col - 5, -1))
    return matrix

matrix = init_matrix(10)
assert matrix[1,2] == 1
assert matrix[1,6] == 12
assert matrix[2,5] == 5
assert matrix[2,6] == 15
assert matrix[3,6] == 1

def ways(max_unit):
    matrix = init_matrix(max_unit)
    return np.sum(matrix, axis=0)[max_unit] + 1

assert ways(5) == 15
print(ways(50))


