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

def sum_proper_devisors(num):
    if num <= 1:
        return 0
    f = prime_factors(num)
    product = 1
    for prime, val in f.items():
        sub_total = 0
        for i in range(0, val+1):
            sub_total += prime ** i
        product *= sub_total
    return product - num



def find(num):
    i = 0
    a = sum_proper_devisors(num)
    while i < 180:
        i += 1
        a = sum_proper_devisors(a)
        print(a)
        if a <= 1:
            break
    return a > 1

find(2862)

my_dict = {0:0, 1:0}
for i in range(2, 1000000):
    sum_divisors = sum_proper_devisors(i)
    if sum_divisors in my_dict:
        my_dict[i] = 0
    else:
        length = 1
        while sum_divisors <= 1000000 and sum_divisors not in my_dict and sum_divisors != i and length <= 180:
            #my_dict[sum_divisors] = 0
            sum_divisors = sum_proper_devisors(sum_divisors)
            length += 1
        if sum_divisors > 1000000 or sum_divisors in my_dict:
            my_dict[i] = 0
        elif length > 50:
            my_dict[i] = 0
            #print('Length > 50 for {}'.format(i))
        else:
            my_dict[i] = length
            print('Found {}, length: {}'.format(i, length))
    if i % 10000 == 0:
        print('Processed {}%'.format(i/10000))

print(my_dict)


for i in range(2,1000000):
    o = sum_proper_devisors(i)
    if i % 10000 == 0:
        print ('processed, ',i)
