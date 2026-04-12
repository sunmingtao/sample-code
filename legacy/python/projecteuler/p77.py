from sieve_prime import get_all_primes
from bisect import bisect_right

all_primes = get_all_primes(50000)

def find_max_prime(n):
    return all_primes[bisect_right(all_primes, n) - 1]

def ways(n, p):
    if n == 0:
        return 1
    if n == 1:
        return 0
    if p > n:
        return ways(n, find_max_prime(n))
    else:
        if p == 2:
            if n % 2 == 0:
                return 1
            else:
                return 0
        else:
            total = 0
            temp_prime = p
            while temp_prime > 2:
                total += sub_ways(n, temp_prime)
                temp_prime = find_max_prime(temp_prime - 1)
            if n % 2 == 0:
                total += 1
            return total


def sub_ways(n, p):
    q = n // p
    total = 0
    remainder = n - p * q
    while q >= 1:
        total += ways(remainder, find_max_prime(p - 1))
        q -= 1
        remainder = n - p * q
    return total


def real_ways(n):
    if n in all_primes:
        return ways(n, n)
    else:
        return ways(n, n+1)

for i in range(2, 100):
    print(i, real_ways(i))


