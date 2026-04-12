import math
import numpy as np
import gmpy2
import time

now = time.time()

def count_reversible_by_digits(n):
    if n % 4 == 1:
        return 0
    if n % 2 == 0:
        return 30 ** (n // 2 - 1) * 20
    else:
        return 20 ** ((n + 1) // 4) * 25 ** ((n + 1) // 4 - 1) * 5

assert count_reversible_by_digits(1) == 0
assert count_reversible_by_digits(2) == 20
assert count_reversible_by_digits(3) == 100
assert count_reversible_by_digits(4) == 600
assert count_reversible_by_digits(5) == 0
assert count_reversible_by_digits(6) == 18000
assert count_reversible_by_digits(7) == 50000
assert count_reversible_by_digits(8) == 540000
assert count_reversible_by_digits(9) == 0
assert count_reversible_by_digits(10) == 30 ** 4 * 20
assert count_reversible_by_digits(11) == 25 ** 2 * 20 ** 3 * 5

print (sum(count_reversible_by_digits(i) for i in range(1, 10)))

print('time spent is {}'.format(time.time() - now))