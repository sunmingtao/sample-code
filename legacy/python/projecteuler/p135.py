import math
import numpy as np
import time

now = time.time()

limit = 10 ** 6

def init_n_dict(limit):
    n_dict = {i:0 for i in range(1, limit)}
    k_limit = (limit - 1) // 4
    for k in range(1, k_limit + 1):
        for a in range(2 * k - 1, -1, -1):
            n = 4 * (k ** 2) - a ** 2
            if n >= limit:
                break
            if a != 0 and k > a:
                n_dict[n] += 2
            else:
                n_dict[n] += 1
    return n_dict

n_dict = init_n_dict(limit)

print (sum(True for _, v in n_dict.items() if v == 10))

print ('time spent is {}'.format(time.time() - now))