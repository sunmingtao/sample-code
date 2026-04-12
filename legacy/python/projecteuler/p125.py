import math
import numpy as np

def sequence_square_start(n, start):
    total = 0
    item_num = 0
    for i in range(start, 0, -1):
        total += i ** 2
        item_num += 1
        if total == n and item_num > 1:
            return True
        elif total > n:
            return False
    return False

assert sequence_square_start(595, 12)
assert not sequence_square_start(1, 1)
assert not sequence_square_start(121, 11)
assert not sequence_square_start(595, 11)

def sequence_square(n):
    start = int(math.sqrt(n))
    for i in range(start, 0, -1):
        if sequence_square_start(n, i):
            return True
    return False

assert sequence_square(595)
assert not sequence_square(959)
assert not sequence_square(1)
assert not sequence_square(121)

def reverse(num):
    num_str = str(num)
    return int(num_str[::-1])


def is_palindrome(num):
    return num == reverse(num)

assert is_palindrome(595)
assert is_palindrome(959)
assert not is_palindrome(956)

total = 0
for i in range (1, 1001):
    if is_palindrome(i) and sequence_square(i):
        total += i

assert total == 4164

total = 0
for i in range (1, 10 ** 8):
    if is_palindrome(i) and sequence_square(i):
        total += i
    if i % 10000 == 0:
        print ('processed {}'.format(i))