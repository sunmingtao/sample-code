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

for n in range(1000001, 1000200, 2):
    if n % 5 != 0:
        an = a(n)
        if an > 1000000:
            print (n, an)
            break



