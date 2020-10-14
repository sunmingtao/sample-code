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

all_primes = get_all_primes(100000)
len(all_primes)


def factor_n(p):
    if p == 2 or p == 5:
        return -1
    rn = 11
    i = 2
    while True:
        remainder = rn % p
        if remainder == 0:
            return i
        rn = remainder * 10 + 1
        i += 1
    return -1

assert factor_n(76801)  == 3200


def isPowerOf(n, base):
    if n == 0:
        return False
    while n != 1:
        if n % base != 0:
            return False
        n = n // base
    return True

assert isPowerOf(1, 2)
assert isPowerOf(2, 2)
assert isPowerOf(32, 2)
assert not isPowerOf(3, 2)
assert not isPowerOf(42, 2)
assert isPowerOf(256, 2)
assert isPowerOf(1024, 2)
assert not isPowerOf(257, 2)
assert isPowerOf(5, 5)
assert isPowerOf(25, 5)
assert isPowerOf(125, 5)
assert not isPowerOf(35, 5)

def remove_trailing_zeros(n):
    if n == 0:
        return 0
    while n % 10 == 0:
        n //= 10
    return n

assert remove_trailing_zeros(0)  == 0
assert remove_trailing_zeros(100)  == 1
assert remove_trailing_zeros(1)  == 1
assert remove_trailing_zeros(120)  == 12

def is_factor(n):
    n = remove_trailing_zeros(n)
    return isPowerOf(n, 2) or isPowerOf(n, 5)

non_factor_list = []
factor_list = []
for prime in all_primes:
    num = factor_n(prime)
    if num > 0 and is_factor(num):
        factor_list.append(prime)
        print (num, prime)
    else:
        non_factor_list.append(prime)


print (len(non_factor_list), len(factor_list))
print (sum(non_factor_list))

print ('time spent is {}'.format(time.time() - now))