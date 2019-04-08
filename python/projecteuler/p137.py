import math
import numpy as np
import gmpy2
import time

now = time.time()

def x(n):
    return 5 * n ** 2 + 2 * n + 1

prev_i = 1
i = 1
count = 0
while True:
    xi = x(i)
    if gmpy2.is_square(xi):
        print (i, xi, i / prev_i)
        prev_i = i
        i = int(i * 6.8541)
        count += 1
        if count == 15:
            break
    else:
        i += 1



print ('time spent is {}'.format(time.time() - now))

