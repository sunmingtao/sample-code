import math
import gmpy2
import time
import sympy
import numpy as np

now = time.time()

def s(k):
    if 1 <= k <= 55:
        return (100003 - 200003 * k + 300007 * (k ** 3)) % 1000000 - 500000
    else:
        return (s(k-24) + s(k-55) + 1000000) % 1000000 - 500000

assert s(10) == -393027
assert s(100) == 86613

for k in range(1, 101):
    print (k, s(k))

def init():
    result = []
    for k in range(1, 4000001):
        if 1 <= k <= 55:
            result.append((100003 - 200003 * k + 300007 * (k ** 3)) % 1000000 - 500000)
        else:
            result.append((result[k-25] + result[k-56] + 1000000) % 1000000 - 500000)
    return result

result = init()
assert result[9] == -393027
assert result[99] == 86613
np_result = np.array(result).reshape(2000,2000)
assert np_result[0, 9] == -393027
assert np_result[0, 99] == 86613
assert np_result[0, 1999] == 33039
assert np_result[1999, 1999] == 442141

def find_greatest_length(arr):
    max_block = []
    temp_total = 0
    temp_max = float("-inf")
    for i in arr:
        temp_total += i
        if temp_total > temp_max:
            temp_max = temp_total
        if temp_total <= 0:
            max_block.append(temp_max)
            temp_total = 0
            temp_max = float("-inf")
    if not math.isinf(temp_max):
        max_block.append(temp_max)
    return max(max_block)

assert find_greatest_length([6, -4, -4, 7, -8, 88, -87, -13, 14, -5]) == 88
assert find_greatest_length([6, -4, -4, 7, -8, 88, -87, -13, 14, -15]) == 88

matrix_len = len(np_result)
max_row = max(find_greatest_length(np_result[i]) for i in range(matrix_len))
max_column = max(find_greatest_length(np_result.transpose()[i]) for i in range(matrix_len))
max_diagonal = max(find_greatest_length(np.diag(np_result, i)) for i in range(1 - matrix_len, matrix_len))
max_antidiagonal = max(find_greatest_length(np.diag(np_result[:, ::-1], i)) for i in range(1 - matrix_len, matrix_len))

print(max(max_row, max_column, max_diagonal, max_antidiagonal))

print('time spent is {}'.format(time.time() - now))
