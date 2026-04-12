import numpy as np

increase_matrix = np.zeros((10, 100), dtype=np.int64)
decrease_matrix = np.zeros((10, 100), dtype=np.int64)

for digit in range(1, 100):
    for n in range(0, 10):
        if digit == 1:
            increase_matrix[n, digit] = 10 - n
            decrease_matrix[n, digit] = n + 1
        else:
            increase_matrix[n, digit] = sum(increase_matrix[i, digit - 1] for i in range(n, 10))
            decrease_matrix[n, digit] = sum(decrease_matrix[i, digit - 1] for i in range(0, n+1))

assert increase_matrix[1,  1] == 9
assert increase_matrix[9,  1] == 1
assert increase_matrix[1,  2] == 45
assert decrease_matrix[0,  1] == 1
assert decrease_matrix[9,  1] == 10
assert decrease_matrix[9,  2] == 55


def num_total_non_bouncy(n, digits):
    return increase_matrix[n, digits] + decrease_matrix[n, digits] - 1 # 111111 counts both an increasing number and a decreasing number


assert num_total_non_bouncy(1, 1) == 10
assert num_total_non_bouncy(9, 1) == 10
assert num_total_non_bouncy(1, 2) == 47
assert num_total_non_bouncy(9, 2) == 55

def num_total_non_bouncy_by_digits(digits):
    if digits == 0:
        return 10
    else:
        return sum(num_total_non_bouncy(i, digits) for i in range(1, 10))

assert num_total_non_bouncy_by_digits(0) == 10
assert num_total_non_bouncy_by_digits(1) == 90
assert num_total_non_bouncy_by_digits(2) == 375


def num_total_non_bouncy_up_to_digits(digits):
    return sum(num_total_non_bouncy_by_digits(i) for i in range(0, digits+1)) - 1 #0000000 is not a number

assert num_total_non_bouncy_up_to_digits(0) == 9
assert num_total_non_bouncy_up_to_digits(2) == 474
assert num_total_non_bouncy_up_to_digits(5) == 12951
assert num_total_non_bouncy_up_to_digits(9) == 277032

print (num_total_non_bouncy_up_to_digits(99))
