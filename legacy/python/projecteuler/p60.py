import math

upper_bound = 100000000
partner_bound = 9999
partner_primes = [i for i in all_primes if i < partner_bound]

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

a = []
for j in [i for i in range(3, 100) if is_prime(i)]:
    a.append([j])
a




good_primes_list_1 = []
for j in [i for i in all_primes if i < 1000]:
    good_primes_list_1.append([j])

print(good_primes_list_1)

prime_limit = {1:9999, 2:9999, 3:9999, 4:9999}

def get_prime_set_list(old_prime_set):
    limit = prime_limit[len(old_prime_set)]
    new_prime_set_list = []
    for i in partner_primes:
        if i>limit:
            break
        bad = False
        for j in old_prime_set:
            if index(all_primes, int(str(j) + str(i))) == -1 or index(all_primes, int(str(i) + str(j))) == -1 or i <= j:
                bad = True
                break
        if not bad:
            old_prime_set_copy = old_prime_set.copy()
            old_prime_set_copy.append(i)
            new_prime_set_list.append(old_prime_set_copy)
    return new_prime_set_list


good_primes_list_2 = []
for i in good_primes_list_1:
    good_primes_list_2.extend(get_prime_set_list(i))

print(good_primes_list_2)

good_primes_list_3 = []
for i in good_primes_list_2:
    good_primes_list_3.extend(get_prime_set_list(i))

print(good_primes_list_3)

good_primes_list_4 = []
for i in good_primes_list_3:
    good_primes_list_4.extend(get_prime_set_list(i))

print(good_primes_list_4)

good_primes_list_5 = []
for i in good_primes_list_4:
    good_primes_list_5.extend(get_prime_set_list(i))

print(good_primes_list_5)