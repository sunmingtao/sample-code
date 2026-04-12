import math

upper_bound = 1000000

from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True

all_primes = [i for i in range(2, upper_bound) if is_prime(i)]

len(all_primes)

consecutive = 21

for c in range(1000, 2, -1):
    found = False
    for i in range(len(all_primes)-c):
        sum_ = sum(all_primes[i:i+c])
        if index(all_primes, sum_) != -1:
            print(c, all_primes[i], sum_)
            found = True
            break
    if found:
        break

all_primes[i:i+consecutive]