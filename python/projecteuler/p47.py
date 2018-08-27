import math


upper_bound = 1000000

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True


all_primes = [i for i in range(2, upper_bound) if is_prime(i)]

def prime_factors(num):
    f = set()
    remaining = num
    for p in all_primes:
        while remaining % p == 0:
            f.add(p)
            remaining /= p
        if remaining == 1:
            break
    return f


prime_factors(644)
prime_factors(645)
prime_factors(646)



i = 200
while i < upper_bound:
    leng = len(prime_factors(i))
    if (leng == 4 and len(prime_factors(i+1)) == 4 and len(prime_factors(i+2)) == 4 and len(prime_factors(i+3)) == 4):
        print(i, leng)
        break
    else:
        i += 1


