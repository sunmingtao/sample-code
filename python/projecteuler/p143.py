import math
import numpy as np
import gmpy2
import time

now = time.time()

def get_opposite_side_square(a,b):
    return a ** 2 + b ** 2 + a * b


pq_dict = {}
limit = 120000
n_limit = int(limit // 4) + 1
for n in range (1, n_limit):
    n3power2 = 3 * n ** 2
    for i in range (int(math.sqrt(3) * n), 0, -1):
        if n3power2 % i == 0:
            p = i + 2 * n
            q = n3power2 // i + 2 * n
            if p + q >= limit:
                break
            if p in pq_dict:
                pq_dict[p].append(q)
            else:
                pq_dict[p] = [q]


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


pqr_set = set()
for p, qs in pq_dict.items():
    for q in qs:
        if q in pq_dict:
            rs = intersection(pq_dict[q], qs)
            if len(rs) > 0:
                for r in rs:
                    if p+q+r <= limit:
                        pqr_set.add(p+q+r)

print (sum(pqr_set))

print('time spent is {}'.format(time.time() - now))

