import math
from bisect import bisect_right
from sieve_prime import get_all_primes

upper_bound = 50000000

p4_bound = int(upper_bound ** (1/4))

p3_bound = int(upper_bound ** (1/3))

p2_bound = int(upper_bound ** (1/2))

p2_primes = get_all_primes(p2_bound)
p3_index = bisect_right(p2_primes, p3_bound)
p4_index = bisect_right(p2_primes, p4_bound)


p3_primes = p2_primes[:p3_index]
p4_primes = p2_primes[:p4_index]


print(len(p2_primes), len(p3_primes), len(p4_primes))

sum_set = set()
for p2 in p2_primes:
    for p3 in p3_primes:
        for p4 in p4_primes:
            total = p2 ** 2 + p3 ** 3 + p4 ** 4
            if total <= upper_bound:
                sum_set.add(total)
            else:
                break

len(sum_set)






