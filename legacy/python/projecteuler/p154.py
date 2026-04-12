import math
import gmpy2
import time
import sympy
import numpy as np
from functools import reduce

import operator as op
from functools import reduce


now = time.time()


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

UPPER_LIMIT = 200000

def find_factor_num(num, factor):
    count = 0
    i = factor
    while num / i >= 1:
        count += int(num / i)
        i *= factor
    return count


assert find_factor_num(28, 2) == 25
assert find_factor_num(11, 5) == 2
assert find_factor_num(28, 5) == 6


def find_trailing_zero(n, c):
    count_2s = find_factor_num(n, 2) - find_factor_num(n - c, 2) - find_factor_num(c, 2)
    count_5s = find_factor_num(n, 5) - find_factor_num(n - c, 5) - find_factor_num(c, 5)
    return min(count_2s, count_5s)


assert find_trailing_zero(28, 0) == 0
assert find_trailing_zero(28, 4) == 0
assert find_trailing_zero(28, 28) == 0
assert find_trailing_zero(100000, 7944) == 6

def find_factor_2_5(n, c):
    count_2s = find_factor_num(n, 2) - find_factor_num(n - c, 2) - find_factor_num(c, 2)
    count_5s = find_factor_num(n, 5) - find_factor_num(n - c, 5) - find_factor_num(c, 5)
    return count_2s, count_5s

assert find_factor_2_5(28, 0) == (0,0)
assert find_factor_2_5(28, 4) == (0,2)
assert find_factor_2_5(28, 28) == (0,0)
assert find_factor_2_5(100000, 7944) == (9,6)
assert find_factor_2_5(156250, 100000)

def find_factor_5(n, c):
    return find_factor_num(n, 5) - find_factor_num(n - c, 5) - find_factor_num(c, 5)




factor_map_5 = {}
for i in range(0, 200001):
    factor_map_5[i] = find_factor_num(i, 5)

factor_map_2 = {}
for i in range(0, 200001):
    factor_map_2[i] = find_factor_num(i, 2)


def calculate_gap_count_2(single_gap_map, gap5, gap2, power):
    total = 0
    for k, v in single_gap_map.items():
        k_gap5, k_gap2 = k
        if k_gap5 + gap5 >= power and k_gap2 + gap2 >= power:
            total += v
    return total


def update_gap_map_2(n, gap_map, power, max_power):
    print ('processing {}'.format(n))
    gap_map[n] = {}
    for r in range(0, n // 2 + 1):
        is_middle = n % 2 == 0 and r == n // 2
        gap5 = factor_map_5[n] - factor_map_5[r] - factor_map_5[n - r]
        gap2 = factor_map_2[n] - factor_map_2[r] - factor_map_2[n - r]
        if gap5 >= power - max_power:
            gap_key = (gap5, gap2)
            if gap_key not in gap_map[n]:
                gap_map[n][gap_key] = 0
            if is_middle:
                gap_map[n][gap_key] += 1
            else:
                gap_map[n][gap_key] += 2
            if r in gap_map:
                gap_map[-1] += calculate_gap_count_2(gap_map[r], gap5, gap2, power)
            if n - r in gap_map and (not is_middle):
                gap_map[-1] += calculate_gap_count_2(gap_map[n - r], gap5, gap2, power)
    gap_map[-1] += calculate_gap_count_2(gap_map[n], 0, 0, power)


now = time.time()
gap_map = {}
gap_map[-1] = 0
upper_limit = 200000
for n in range(0, upper_limit + 1):
    update_gap_map_2(n, gap_map, 12, 7)
print ('solution={}'.format(gap_map[-1]))
print('time spent is {}'.format(time.time() - now))

def count_power(upper_limit, power):
    total = 0
    for n in range(1, upper_limit + 1):
        for i in range(0, n+1):
            for j in range(n, upper_limit+1):
                gap5 = factor_map_5[j] - factor_map_5[n - i] - factor_map_5[i] - factor_map_5[j - n]
                gap2 = factor_map_2[j] - factor_map_2[n - i] - factor_map_2[i] - factor_map_2[j - n]
                if gap5 >= power and gap2 >= power:
                    total += 1
    return total

print (count_power(100, 4))


def count_power_2(upper_limit, power):
    total = 0
    for n in range(1, upper_limit+1):
        print ('processing {}'.format(n))
        for i in range(0, n // 2 + 1):
            gap5 = factor_map_5[upper_limit] - factor_map_5[upper_limit - n] - factor_map_5[n - i] - factor_map_5[i]
            gap2 = factor_map_2[upper_limit] - factor_map_2[upper_limit - n] - factor_map_2[n - i] - factor_map_2[i]
            if gap5 >= power and gap2 >= power:
                if n % 2 == 0 and i == n // 2:
                    total += 1
                else:
                    total += 2
    return total

assert count_power_2(5, 1) ==12
assert count_power_2(6, 1) ==13
assert count_power_2(7, 1) ==6
assert count_power_2(8, 1) ==15
assert count_power_2(100, 2) == 4881
assert count_power_2(100, 3) == 2121
assert count_power_2(100, 4) == 480
assert count_power_2(100, 5) == 0
assert count_power_2(1000, 3) == 488814

now = time.time()
print ('solution={}'.format(count_power_2(200000, 12)))
print('time spent is {}'.format(time.time() - now))


def count_power_3(upper_limit, power):
    total = 0
    i = upper_limit
    while i > 0:
        start = upper_limit - i
        end = upper_limit - start * 2
        if start > end:
            break
        end2 = end
        if end == start:
            end2 += 1
        for j in range(start, end2):
            gap5 = factor_map_5[upper_limit] - factor_map_5[upper_limit - i] - factor_map_5[i - j] - factor_map_5[j]
            gap2 = factor_map_2[upper_limit] - factor_map_2[upper_limit - i] - factor_map_2[i - j] - factor_map_2[j]
            if gap5 >= power and gap2 >= power:
                if end == start:
                    total += 1
                else:
                    total += 3
        i -= 1
    return total

print (count_power_3(200000, 12))


def count_power_4(upper_limit, power):
    max_power = int(math.log(upper_limit, 5))
    total = 0
    i = upper_limit
    while i > 0:
        print ('Processing {}'.format(i))
        gap5_1 = factor_map_5[upper_limit] - factor_map_5[upper_limit - i] - factor_map_5[i]
        if gap5_1 >= power - max_power:
            gap2_1 = factor_map_2[upper_limit] - factor_map_2[upper_limit - i] - factor_map_2[i]
            start = upper_limit - i
            end = upper_limit - start * 2
            if start > end:
                break
            end2 = end
            if end == start:
                end2 += 1
            for j in range(start, end2):
                gap5 = gap5_1 + factor_map_5[i] - factor_map_5[i - j] - factor_map_5[j]
                gap2 = gap2_1 + factor_map_2[i] - factor_map_2[i - j] - factor_map_2[j]
                if gap5 >= power and gap2 >= power:
                    if end == start:
                        total += 1
                    else:
                        total += 3
        i -= 1
    return total


assert count_power_4(5, 1) ==12
assert count_power_4(6, 1) ==13
assert count_power_4(7, 1) ==6
assert count_power_4(8, 1) ==15
assert count_power_4(100, 2) == 4881
assert count_power_4(100, 3) == 2121
assert count_power_4(100, 4) == 480
assert count_power_4(100, 5) == 0
assert count_power_4(1000, 3) == 488814
assert count_power_4(1000, 5) == 246960

now = time.time()
print (count_power_4(200000, 12))
print('time spent is {}'.format(time.time() - now))


def count_power_5(upper_limit, power):
    iset = set()
    max_power = int(math.log(upper_limit, 5))
    total = 0
    i = upper_limit
    while i > 0:
        print ('Processing {}'.format(i))
        gap5 = factor_map_5[upper_limit] - factor_map_5[upper_limit - i] - factor_map_5[i]
        if gap5 >= power - max_power:
            gap2 = factor_map_2[upper_limit] - factor_map_2[upper_limit - i] - factor_map_2[i]
            start = upper_limit - i
            end = upper_limit - start * 2
            if start > end:
                break
            for j in range(start, i // 2 + 1):
                new_gap5 = gap5 + factor_map_5[i] - factor_map_5[i - j] - factor_map_5[j]
                new_gap2 = gap2 + factor_map_2[i] - factor_map_2[i - j] - factor_map_2[j]
                if new_gap5 >= power and new_gap2 >= power:
                    if i not in iset:
                        print (i,j, i-j)
                        iset.add(i)
                    if end == start:
                        total += 1
                    elif j == start:
                        total += 3
                    elif i % 2 == 0 and j == i // 2:
                        total += 3
                    else:
                        total += 6
        i -= 1
        if i == upper_limit - 1000:
            break
    return total


assert count_power_5(5, 1) ==12
assert count_power_5(6, 1) ==13
assert count_power_5(7, 1) ==6
assert count_power_5(8, 1) ==15
assert count_power_5(100, 2) == 4881
assert count_power_5(100, 3) == 2121
assert count_power_5(100, 4) == 480
assert count_power_5(100, 5) == 0
assert count_power_5(1000, 3) == 488814
assert count_power_5(1000, 5) == 246960

now = time.time()
print (count_power_5(200000, 12))
print('time spent is {}'.format(time.time() - now))

def count_power_6(upper_limit, power):
    total = 0
    for a in range (0, upper_limit // 3 + 1):
        print ('Processing a = {}'.format(a))
        for b in range(a, (upper_limit - a) // 2 + 1):
            c = upper_limit - a - b
            if c < b:
                break
            gap5 = factor_map_5[upper_limit] - factor_map_5[a] - factor_map_5[b] - factor_map_5[c]
            gap2 = factor_map_2[upper_limit] - factor_map_2[a] - factor_map_2[b] - factor_map_2[c]
            if gap5 >= power and gap2 >= power:
                if a == b == c:
                    total += 1
                elif a == b or b == c:
                    total += 3
                else:
                    total += 6
    return total

assert count_power_6(5, 1) ==12
assert count_power_6(6, 1) ==13
assert count_power_6(7, 1) ==6
assert count_power_6(8, 1) ==15
assert count_power_6(100, 2) == 4881
assert count_power_6(100, 3) == 2121
assert count_power_6(100, 4) == 480
assert count_power_6(100, 5) == 0
assert count_power_6(1000, 3) == 488814
assert count_power_6(1000, 5) == 246960

now = time.time()
print (count_power_6(200000, 12))
print('time spent is {}'.format(time.time() - now))




