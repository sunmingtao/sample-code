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

all_primes = get_all_primes(1000003)
print(len(all_primes))


def find_n(small_p, big_p):
    small_p_str = str(small_p)
    big_p_str = str(big_p)
    n = ''
    for i in range(0, len(small_p_str)):
        small_i = len(small_p_str) - i - 1
        big_i = len(big_p_str) - i - 1
        for j in range(0, 10):
            if ((int(big_p_str[big_i:]) * int(str(j) + n)) - int(small_p_str[small_i:])) % (10 ** (i+1)) == 0:
                n = str(j) + n
                break
    return int(n)

assert find_n(19, 23) == 53
assert find_n(999983, 1000003) == 666661



total = 0
for small_p, big_p, index in zip(all_primes[2:-1], all_primes[3:], range(0, len(all_primes[3:]))):
    n = find_n(small_p, big_p)
    total += big_p * n

print(total)


print ('time spent is {}'.format(time.time() - now))
