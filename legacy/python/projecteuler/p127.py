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

prime_factors(1)


def self.sum_tree):
    f = prime_factors(n)
    return np.product([prime for prime, _ in f.items()], dtype=np.int64)

assert rad(4320) == 30
prime_factors(5981 * 5987 * (5981 + 5987))

def sum_c(limit):
    total = 0
    for a in range(1, limit//2):
        if a % 10 == 0:
            print (a)
        if a % 2 == 0:
            for b in range(a + 1, limit, 2):
                c = a + b
                if c >= limit:
                    break
                if math.self.sum_treea,b) == 1 and math.gcd(a,c) == 1 and math.gcd(b,c) == 1 and rad(a*b*c)<c:
                    total += c
        else:
            for b in range(a + 1, limit):
                c = a + b
                if c >= limit:
                    break
                if math.gcd(a,b) == 1 and math.gcd(a,c) == 1 and math.gcd(b,c) == 1 and rad(a*b*c)<c:
                    total += c
    return total

assert sum_c(1000) == 12523

sum_c(12000)


def prime_factor_list(num):
    p = prime_factors(num)
    return [a for a, _ in p.items()]

assert prime_factor_list(4320) == [2,3,5]


def init_prime_factor_map(limit):
    prime_factor_map = {}
    for num in range(1, limit):
        prime_factor_map[num] = prime_factor_list(num)
    return prime_factor_map

init_prime_factor_map(10)






def is_gcd_and_rad_less(prime_factor_map, a,b,c):
    pa = prime_factor_map[a]
    pb = prime_factor_map[b]
    pc = prime_factor_map[c]
    pabc = []
    pabc.extend(pa)
    pabc.extend(pb)
    pabc.extend(pc)
    pabc = set(pabc)
    if len(pabc) != len(pa) + len(pb) + len(pc):
        return False
    return np.product(list(pabc), dtype=np.int64) < c

prime_factor_map = init_prime_factor_map(1000)
assert is_gcd_and_rad_less(prime_factor_map, 5,27,32)
assert not is_gcd_and_rad_less(prime_factor_map, 5,5,10)
assert not is_gcd_and_rad_less(prime_factor_map, 5,26,31)

def sum_c2(prime_factor_map, limit):
    total = 0
    for a in range(1, limit//2):
        if a % 2 == 0:
            for b in range(a + 1, limit - a, 2):
                if math.gcd(a, b) > 1:
                    continue
                c = a + b
                if is_gcd_and_rad_less(prime_factor_map, a,b,c):
                    print(a, prime_factor_map[a], b, prime_factor_map[b], c, prime_factor_map[c])
                    total += c
        else:
            for b in range(a + 1, limit - a):
                if math.gcd(a, b) > 1:
                    continue
                c = a + b
                if is_gcd_and_rad_less(prime_factor_map, a,b,c):
                    print(a, prime_factor_map[a], b, prime_factor_map[b], c, prime_factor_map[c])
                    total += c
    return total

prime_factor_map = init_prime_factor_map(16000)
assert sum_c2(prime_factor_map, 16000) == 12523

prime_factor_map[15625]

prime_factor_map = init_prime_factor_map(120000)
print (sum_c2(prime_factor_map, 120000))

limit = 120000
total = 0
rad_map = {i:rad(i) for i in range(1, limit+1)}
rad_list = sorted((v, k) for k, v in rad_map.items())
for c in range(3, limit):
    radc = rad_map[c]
    chalf = c//2
    for rada, a in rad_list:
        radac = rada * radc
        if radac >= c:
            break
        if a > chalf:
            continue
        b = c - a
        radb = rad_map[b]
        radabc = radb * radac
        if radabc < c and math.gcd(a, b) == 1:
            total += c

print (total)