import math
import numpy as np
import gmpy2
import time

now = time.time()

def x(n):
    return 5 * n ** 2 + 14 * n + 1

multiply_odd = 2.5
multiply_even = 4.2
prev_i = 1
i = 2
multiply = 0
golden_nuggets = []
while len(golden_nuggets) < 30:
    xi = x(i)
    if gmpy2.is_square(xi):
        temp_i = i
        print (temp_i, xi, temp_i / prev_i)
        golden_nuggets.append(temp_i)
        if len(golden_nuggets) % 2 == 0:
            i = int(temp_i * multiply_even)
            multiply_odd = temp_i / prev_i
        else:
            i = int(temp_i * multiply_odd)
            if prev_i > 1:
                multiply_even = temp_i / prev_i
        prev_i = temp_i
    else:
        i -= 1

print (sum(golden_nuggets))

print ('time spent is {}'.format(time.time() - now))

