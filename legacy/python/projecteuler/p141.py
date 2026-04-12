import math
import numpy as np
import gmpy2
import time

now = time.time()


n_list = []
limit = 10 ** 12
for a in range (1, 10 ** 4):
    for b in range (1, a):
        if a ** 3 * b + b ** 2 >= limit:
            break
        if math.gcd(a,b) > 1:
            continue
        k = 1
        while True:
            n = a ** 3 * k ** 2 * b + k * b ** 2
            if n >= limit:
                break
            if gmpy2.is_square(n):
                n_list.append(n)
            k += 1

print (sum(n_list))

print ('time spent is {}'.format(time.time() - now))