import numpy as np
import math


def init_matrix(round):
    matrix = np.zeros((round + 1, round + 1), dtype=np.long)
    for row in range(1, round+1):
        for col in range(0, round+1):
            if row == col:
                matrix[row, col] = 1
            elif col == 0:
                if row == 1:
                    matrix[row, col] = 1
                else:
                    matrix[row, col] = matrix[row - 1, col] * row
            else:
                matrix[row, col] = matrix[row - 1, col - 1] + matrix[row - 1, col] * row
    return matrix

matrix = init_matrix(4)
assert matrix[4,3] == 10
assert matrix[4,2] == 35
assert matrix[4,1] == 50
assert matrix[4,4] == 1

def win_num(round):
    matrix = init_matrix(round)
    win_index = round // 2 + 1
    return sum(matrix[round][win_index:])

assert win_num(4) == 11

def max_payout(round):
    total_num = math.factorial(round+1)
    wins = win_num(round)
    return total_num // wins

assert max_payout(4) == 10
assert max_payout(1) == 2

print (max_payout(15))