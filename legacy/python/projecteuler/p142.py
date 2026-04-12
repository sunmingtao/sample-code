import math
import numpy as np
import gmpy2
import time

now = time.time()


def find_xyz():
    n = 3
    while True:
        for p in range(2, n):
            if not gmpy2.is_square(n ** 2 - p ** 2):
                continue
            for m in range(1, p):
                if p ** 2 <= (n ** 2 + m ** 2) / 2:
                    break
                if int(math.fabs(n - m)) % 2 == 1:
                    continue
                nmp = n ** 2 + m ** 2 - p ** 2
                if gmpy2.is_square(nmp) and gmpy2.is_square(p ** 2 - m ** 2):
                    x = (n ** 2 + m ** 2) // 2
                    y = (n ** 2 - m ** 2) // 2
                    z = p ** 2 - x
                    return x+y+z
        n += 1

print (find_xyz())

print ('time spent is {}'.format(time.time() - now))