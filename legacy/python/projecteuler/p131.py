import math
import numpy as np


def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True


limit = 1000000

total = 0
for power in range (1, 577):
    n = power ** 3
    k = n + power ** 2
    p = 3 * power ** 2 + 3 * power + 1
    if is_prime(p):
        total += 1
        print(p, power, n, k)

print(total)





