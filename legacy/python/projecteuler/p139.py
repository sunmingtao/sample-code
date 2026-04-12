import math
import numpy as np
import gmpy2
import time

LIMIT = 100000000
abc_list = []
multiply = 1
prev_a = 3 / 7
a = 3
while True:
    c_square = a ** 2 + (a + 1) ** 2
    if gmpy2.is_square(c_square):
        multiply = a / prev_a
        c = int(math.sqrt(c_square))
        abc = 2*a+1+c
        if abc < LIMIT:
            abc_list.append(abc)
        else:
            break
        print(a, 2*a+1+c, multiply)
        prev_a = a
        a = int(a * multiply)
        if a >= LIMIT:
            break
    else:
        a -= 1


print (sum(LIMIT // p for p in abc_list))



print ('time spent is {}'.format(time.time() - now))

