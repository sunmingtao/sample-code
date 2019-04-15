import math
import numpy as np
import gmpy2
import time

now = time.time()


def find_square(l):
    return 4 + 5 * (l ** 2 - 1)

l_list = []
i = 3
prev_i = 1
while True:
    square = find_square(i)
    if gmpy2.is_square(square):
        found_i = i
        l_list.append(found_i)
        if len(l_list) == 12:
            break
        i = int(found_i * (found_i / prev_i))
        prev_i = found_i
        if i % 2 == 0:
            i -= 1
    else:
        i += 2

print (sum(l_list))




print ('time spent is {}'.format(time.time() - now))

