import math
from sieve_prime import get_all_primes
from bisect_index import get_index




def phi(n):
    return sum(math.gcd(i, n) == 1  for i in range(1,n))

def prime_factors(num):
    f = set()
    if get_index(all_primes, num) > -1:
        f.add(num)
        return f
    else:
        remaining = num
        for p in all_primes:
            while remaining % p == 0:
                f.add(p)
                remaining //= p
            if remaining == 1:
                return f


def phi2(n):
    result = n
    factors = prime_factors(n)
    for f in factors:
        result *= f - 1
        result //= f
    return result


limit = 1000000
all_primes = get_all_primes(limit)
sum(phi2(i) for i in range(2, limit+1))


