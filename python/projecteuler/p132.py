import math
import numpy as np
import time

now = time.time()


def get_all_primes(n):
    arr = [True for n in range(n+1)]
    arr[0], arr[1] = False, False
    arr[2] = True
    index = 2
    while index < len(arr):
        for j in range(index * 2, n+1, index):
            arr[j] = False
        index += 1
        while index < len(arr) and not arr[index]:
            index += 1
    return [i for i in range(n+1) if arr[i]]

all_primes = get_all_primes(1000000)
len(all_primes)


def factor_n(num, p):
    rn = 11
    for i in range(2, num):
        remainder = rn % p
        if remainder == 0:
            return i
        rn = remainder * 10 + 1
    return -1


prime_list2 = []
for prime in all_primes[1:]:
    num = factor_n(10 ** 6, prime)
    if num > 0 and 10 ** 9 % num == 0:
        prime_list2.append(prime)
        if len(prime_list2) == 40:
            break

sum(prime_list2)

print ('time spent is {}'.format(time.time() - now))