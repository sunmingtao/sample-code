import math
import gmpy2
import time
import sympy
import numpy as np

now = time.time()

def find_max_ij(k):
    return int(math.sqrt(k-1))

assert find_max_ij(12) == 3
assert find_max_ij(10000) == 99
assert find_max_ij(10 ** 8) == 9999

def sum_mod(n):
    half_n = (n + 1) // 2
    return n * n - sum(n % i for i in range(1, n // 2 + 1)) - (half_n * (half_n - 1)) // 2

assert sum_mod(1) == 1
assert sum_mod(2) == 4


sum_mod_cache = {}

def find_imaginary_num_from_root_ij(upper_limit, root_i, root_j):
    num = upper_limit // (root_i ** 2 + root_j ** 2)
    if num not in sum_mod_cache:
        sum_mod_cache[num] = sum_mod(num)
    if root_i == root_j:
        return sum_mod_cache[num] * 2 * root_i
    else:
        return sum_mod_cache[num] * 2 * (root_i + root_j)


assert find_imaginary_num_from_root_ij(10, 1, 1) == 42
assert find_imaginary_num_from_root_ij(12, 1, 1) == 66
assert find_imaginary_num_from_root_ij(13, 1, 1) == 66
assert find_imaginary_num_from_root_ij(14, 1, 1) == 82
assert find_imaginary_num_from_root_ij(5, 1, 1) == 8
assert find_imaginary_num_from_root_ij(5, 2, 1) == 6



def find_imaginary_solution_num(upper_limit):
    imaginary_total = 0
    for i in range(1, find_max_ij(upper_limit) + 1):
        for j in range(1, i+1):
            if math.gcd(i, j) == 1:
                if i ** 2 + j ** 2 > upper_limit:
                    break
                else:
                    imaginary_value = find_imaginary_num_from_root_ij(upper_limit, i, j)
                    imaginary_total += imaginary_value
                    #print (i, j, imaginary_value)
    return imaginary_total

assert find_imaginary_solution_num(5) == 14
assert find_imaginary_solution_num(12) == 98
assert find_imaginary_solution_num(10 ** 5) == 9699916320


def find_rational_solution_num(upper_limit):
    return sum_mod(upper_limit)

assert find_rational_solution_num(5) == 21
assert find_rational_solution_num(10**5) == 8224740835


def find_solution(upper_limit):
    return find_rational_solution_num(upper_limit) + find_imaginary_solution_num(upper_limit)

assert find_solution(5) == 35
assert find_solution(10 ** 5) == 17924657155
print (find_solution(10 ** 8))



print('time spent is {}'.format(time.time() - now))



