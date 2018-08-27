import math
import numpy as np
from sieve_prime import get_all_primes

from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

upper_bound = 900000000

all_primes = get_all_primes(upper_bound)

len(all_primes)

def get_corner(i, j, corners):
    if i == 0:
        return 1 + 8 * i + j * 2 + 2
    else:
        return corners[i-1][j] + 8 * i + j * 2 + 2


def get_ratio(n, corners):

    for i in range(n):
        for j in range(4):
            corners[i][j] = get_corner(i, j, corners)

    corner_digits = np.array(corners).reshape(-1).tolist()
    prime_num = sum(i in all_primes for i in corner_digits)
    return prime_num/(len(corner_digits)+1)


n=20000
corners = [[0 for col in range(4)] for row in range(n)]
for i in range(n):
    for j in range(4):
        corners[i][j] = get_corner(i, j, corners)
corners[19999]

for i in range(13000, 14000):
    corner_digits = np.array(corners[:i+1]).reshape(-1).tolist()
    prime_num = sum(index(all_primes, i) != -1 for i in corner_digits)
    if prime_num/(len(corner_digits)+1) < 0.1:
        print(i+1, prime_num/(len(corner_digits)+1), corners[i])