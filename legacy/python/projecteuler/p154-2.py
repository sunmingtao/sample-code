import math
import gmpy2
import time
import sympy
import numpy as np
from functools import reduce

now = time.time()

n = 3

1,3,6,10,15,21

def print_sequence(n):
    tail = 1
    head = n
    i = 1
    entry = 1
    while i < 120 - n:
        print (entry, entry % 1000)
        head += 1
        entry = entry * head // tail
        tail += 1
        i += 1


print_sequence(100)


def print_sequence2(n):
    tail = 1
    head = n
    i = 1
    entry = 1
    while i < 120 - n:
        head += 1
        print(entry, head, tail, entry % 1000)
        entry = entry * head // tail
        entry %= 1000
        tail += 1
        i += 1





import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def find_zeros(n):
    s = str(n)
    return len(s) - len(s.rstrip('0'))



def find_6_zeros(n, limit):
    for i in range(n, limit):
        ncr_value = ncr(i, n)
        zeros = find_zeros(ncr_value)
        if zeros >= 6:
            print (i, n, zeros)




def find_many_zeros(n):
    for i in range(1, n // 2 + 1):
        ncr_value = ncr(n, i)
        zeros = find_zeros(ncr_value)
        if zeros >= 5:
            print(i, n, zeros)


def find_factor_num(num, factor):
    count = 0
    i = factor
    while (num / i >= 1):
        count += int(num / i)
        i *= factor
    return int(count)


assert find_factor_num(28, 2) == 25
assert find_factor_num(11, 5) == 2
assert find_factor_num(28, 5) == 6

def find_trailing_zero_slow(n, c):
    return find_zeros(ncr(n, c))

def find_trailing_zero(n, c):
    count_2s = find_factor_num(n, 2) - find_factor_num(n - c, 2) - find_factor_num(c, 2)
    count_5s = find_factor_num(n, 5) - find_factor_num(n - c, 5) - find_factor_num(c, 5)
    return min(count_2s, count_5s)

assert find_trailing_zero(28, 4) == 0
assert find_trailing_zero(100000, 7944) == 6
assert find_trailing_zero(83498, 40624) == 7
assert find_trailing_zero(78125, 3126) == 7
assert find_trailing_zero(3126, 23) == 5
assert find_trailing_zero(48827, 24373) == 6
assert find_trailing_zero (24373, 9374) ==6
assert find_trailing_zero(48827, 24454) == 6
assert find_trailing_zero (24454, 9374) ==6

for n in range(1, 1001):
    for c in range(1, n+1):
        assert find_trailing_zero(n, c) == find_trailing_zero_slow(n,c)

print('time spent is {}'.format(time.time() - now))



