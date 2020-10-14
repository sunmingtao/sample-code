import math
import numpy as np
import gmpy2
import time

now = time.time()

def ellipse(x, y):
    return 4 * x ** 2 + y ** 2

assert ellipse(1.4, -9.6) == 100

def find_k2 (k1, k0):
    return (2 * k0 + k1 * k0 ** 2 - k1) / (1 + 2 * k0 * k1 - k0 ** 2)


def find_k (x0, y0, x1, y1):
    return (y0 - y1) / (x0 - x1)

def find_k0 (x, y):
    return -4 * x / y

def find_b(k, x, y):
    return y - k * x

def find_y (x, k, b):
    return x * k + b

def find_hit(k, b):
    sqrt = math.sqrt(4 * k ** 2 * b ** 2 - 4 * (4 + k ** 2) * (b ** 2 - 100))
    x1 = (-2 * k * b + sqrt) / (2 * (4 + k ** 2))
    y1 = k * x1 + b
    x2 = (-2 * k * b - sqrt) / (2 * (4 + k ** 2))
    y2 = k * x2 + b
    return (x1, y1), (x2, y2)


episolon = 0.00001

def find_next_hit(start_x, start_y, end_x, end_y):
    k1 = find_k(start_x, start_y, end_x, end_y)
    k0 = find_k0(end_x, end_y)
    k2 = find_k2(k1, k0)
    b2 = find_b(k2, end_x, end_y)
    (x1, y1), (x2, y2) = find_hit(k2, b2)
    found = 0
    if math.fabs(x1 - end_x) <= episolon and math.fabs(y1 - end_y) <= episolon:
        found += 1
    if math.fabs(x2 - end_x) <= episolon and math.fabs(y2 - end_y) <= episolon:
        found += 1
    assert found == 1
    if math.fabs(x1 - end_x) > episolon and math.fabs(y1 - end_y) > episolon:
        x, y = x1, y1
    else:
        x, y = x2, y2
    assert math.fabs(ellipse(x, y) - 100) < episolon
    return x, y


start_x, start_y, end_x, end_y = 0, 10.1, 1.4, -9.6
hit = 0
while not (-0.01 <= end_x <= 0.01 and end_y > 0):
    next_end_x, next_end_y = find_next_hit(start_x, start_y, end_x, end_y)
    start_x, start_y = end_x, end_y
    end_x, end_y = next_end_x, next_end_y
    hit += 1
    print (end_x, end_y)

print (hit)


print('time spent is {}'.format(time.time() - now))

