import math
import numpy as np

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

def prime_factors(num):
    f = {}
    remaining = num
    for p in all_primes:
        while remaining % p == 0:
            if p in f:
                f[p] += 1
            else:
                f[p] = 1
            remaining /= p
        if remaining == 1:
            break
    return f


prime_factors(504)

def rad(n):
    f = prime_factors(n)
    return np.product([prime for prime, _ in f.items()], dtype=np.int64)

assert rad(504) == 42

limit = 100001
rad_list = []
for i in range (1, limit):
    rad_list.append((rad(i), i))




rad_list.sort()

print(rad_list[9999])

