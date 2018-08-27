import math
from sieve_prime import get_all_primes

all_primes = get_all_primes(10000000)

all_primes[-1]

len(all_primes)


def is_relative_prime(num1, num2):
    return math.gcd(num1, num2) == 1

def phi(n):
    return sum(is_relative_prime(i, n) for i in range(1,n))

def is_perm(num1, num2):
    num1_str = str(num1)
    num2_str = str(num2)
    for c in '0123456789':
        if num1_str.count(c) != num2_str.count(c):
            return False
    return True


#max(i/phi(i) for i in range(2,upper_bound) if i % 2 == 0)
print(17*13)
phi(17*13)

def phi2(m,n):
    return m * n - m - n + 1

ratio = 2
good_n = 1
for i in range(1,700):
    for j in range(1,700):
        n = all_primes[i] * all_primes[j]
        if n >= 10000000:
            break
        phi_ = phi2(all_primes[i],all_primes[j])
        if is_perm(n, phi_):
            new_ratio = n/phi_
            if new_ratio < ratio:
                ratio = new_ratio
                good_n = n

print(good_n, ratio)




