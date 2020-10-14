import math
import numpy as np

def a(n):
    rk = 1
    for k in range(1, 2000000):
        if rk % n == 0:
            return k
        else:
            rk = (rk * 10 + 1) % n
    return 0

assert a(17) == 16
assert a(41) == 5


def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True


total = 0
count = 0
for n in range(9, 100000, 2):
    if n % 5 != 0:
        an = a(n)
        if not is_prime(n) and (n - 1) % an == 0:
            count += 1
            total += n
            if count == 25:
                break

print (total, count)





