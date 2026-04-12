import math

from sieve_prime import get_all_primes

all_primes = get_all_primes(1000000)

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

def get_proper_factors(num):
    f = prime_factors(num)
    product = 1
    for prime, num in f:
        sub_total = 0
        for i in range(0, num+1):
            sub_total += prime ** i
        product *= sub_total
    return product

get_proper_factors(28)

def factorial_sum(num):
    return sum(math.factorial(int(k)) for k in str(num))


for i in range(11, 99999):
    if i == factorial_sum(i):
        print(i)


factorial_sum(0)

