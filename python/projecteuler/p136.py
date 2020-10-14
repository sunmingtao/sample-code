import math
import numpy as np
import time

now = time.time()

limit = 50 * (10 ** 6)

def init_n_dict(limit):
    n_dict = {}
    k_limit = (limit - 1) // 4
    for k in range(1, k_limit + 1):
        n0 = 4 * (k ** 2)
        for a in range(2 * k - 1, -1, -1):
            n = n0 - a ** 2
            if n >= limit:
                break
            if n not in n_dict:
                n_dict[n] = 0
            if a != 0 and k > a:
                n_dict[n] += 2
            else:
                n_dict[n] += 1
    return n_dict

n_dict = init_n_dict(limit)

print (sum(True for _, v in n_dict.items() if v == 1))

print ('time spent is {}'.format(time.time() - now))