from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

twice_square_list = [2 * i ** 2 for i in range(1, 100)]

import math

upper_bound = 10000

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True


all_primes = [i for i in range(2, upper_bound) if is_prime(i)]

all_odd_composite = [i for i in range(3, upper_bound, 2) if not is_prime(i)]

for i in all_odd_composite:
    found = False
    for j in twice_square_list:
        if i - j > 1 and index(all_primes, i - j) != -1:
            found = True
            break
    if not found:
        print(i)





