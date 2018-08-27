import math

upper_bound = 7654321

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True


all_primes = [i for i in range(2, upper_bound) if is_prime(i)]

all_primes.sort(reverse=True)

for p in all_primes:
    if is_pan(p):
        print (p)
        break

def is_pan(n):
    n_str = str(n)
    leng = len(n_str)
    return n_str.find('0') == -1 and int(max(n_str)) == leng and len(set(n_str)) == leng

max('76654321')
is_pan(7654329)


'abc'.find('c')
max('7654319')
