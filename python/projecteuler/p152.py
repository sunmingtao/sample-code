import math
import gmpy2
import time
import sympy
import numpy as np

now = time.time()


a1=[2,3,4,5,7,12,15,20,28,35]
a2=[2,3,4,6,7,9,10,20,28,35,36,45]
a3=[2,3,4,6,7,9,12,15,28,30,35,36,45]

def inverse_square_sum(arr):
    return sum(1 / i ** 2 for i in arr)



inverse_square_sum([2,3,4,5,7,12,15,20,28,35])

inverse_square_sum([2,3,4,5,6,12, 24])
inverse_square_sum([2,3,4,5,6,12, 26, 79, 80])


def find_prod(arr):
    p = 1
    for i in arr:
        p *= i ** 2
    return p


def is_half(a):
    return math.fabs(a - 0.5) <= 10 ** -12

assert is_half(1/2)
assert not is_half(0.4999999)
assert is_half(0.4999999999994)
assert is_half(0.49999999999999994)
assert is_half(0.500000000000005)
assert not is_half(0.5000055245087296)

def less_than_half(a):
    return a < 0.5 and (not is_half(a))

assert not less_than_half(1/2)
assert less_than_half(0.4999999)
assert not less_than_half(0.4999999999994)
assert not less_than_half(0.49999999999999994)


def more_than_half(a):
    return a > 0.5 and (not is_half(a))

assert more_than_half(0.5000055245087296)
assert not more_than_half(1/2)
assert not more_than_half(0.500000000000005)

UPPER_LIMIT = 80
MAGIC_NUMBER = 32760

def find_next_min(arr):
    arr_max = arr[-1]
    s_sum = inverse_square_sum(arr)
    temp_s_sum = s_sum + 1 / (arr_max + 1) ** 2
    if arr[-1] == UPPER_LIMIT or more_than_half(s_sum + 1 / UPPER_LIMIT ** 2):
        return None
    elif is_half(temp_s_sum):
        return arr_max + 1, True
    elif less_than_half(temp_s_sum):
        return arr_max + 1, False
    else:
        lower_bound = arr_max + 1
        upper_bound = UPPER_LIMIT
        candidate = (lower_bound + upper_bound) // 2
        while upper_bound - lower_bound > 1:
            temp_s_sum = s_sum + 1 / candidate ** 2
            #print ('sum is {} for candidate {}'.format(temp_s_sum, candidate))
            if is_half(temp_s_sum):
                return candidate, True
            elif less_than_half(temp_s_sum):
                upper_bound = candidate
                #print('upper bound changes to {}'.format(upper_bound))
                candidate = (lower_bound + upper_bound) // 2
            else:
                lower_bound = candidate
                #print('lower bound changes to {}'.format(lower_bound))
                candidate = (lower_bound + upper_bound) // 2
        #print('lower bound is {}, higher bound is {}, candidate is {}'.format(lower_bound, upper_bound, candidate))
        return upper_bound, False



assert find_next_min([2]) == (3, False)
assert find_next_min([2,3]) == (4, False)
assert find_next_min([2,3,4]) == (5, False)
assert find_next_min([2,3,4,5]) == (6, False)
assert find_next_min([2,3,4,5,6]) == (11, False)
assert find_next_min([2,3,4,5,6,11]) == (54, False)
assert find_next_min([2,3,4,5,6,12]) == (25, False)
assert find_next_min([2,3,4,5,6,11,54]) is None
assert find_next_min([2,3,4,5,7,12,15,20,28]) == (35, True)
assert find_next_min([2,3,4,5,7,12,15,20,28,35]) is None
assert find_next_min([2,3,4,5,6,11,73]) == (80, False)
assert find_next_min([2,3,4,5,6,11,73, 80]) is None
assert find_next_min([2, 3, 4, 5, 6, 11, 80]) is None


def update_arr(arr):
    arr[-1] += 1
    while MAGIC_NUMBER % arr[-1] != 0:
        arr[-1] += 1
    if arr[-1] > UPPER_LIMIT:
        del arr[-1]
        return update_arr(arr)
    else:
        return arr

#assert update_arr([2, 3, 4, 6, 7, 10, 13, 14, 20, 21, 30, 39, 70]) == [2, 3, 4, 6, 7, 10, 13, 14, 20, 21, 30, 39, 77]
#assert update_arr([2, 3, 4, 6, 7, 10, 13, 14, 20, 21, 30, 39, 78]) == [2, 3, 4, 6, 7, 10, 13, 14, 20, 21, 30, 42]


def inverse_square_sum2(arr):
    prod = find_prod(arr)
    s = sum(prod // i ** 2 for i in arr)
    return prod / s

def find_sequence():
    candidate = set()
    total = 0
    arr = [2, 3]
    while arr[0] == 2:
        next_min = find_next_min(arr)
        if next_min is not None:
            next_min_value = next_min[0]
            while MAGIC_NUMBER % next_min_value != 0:
                next_min_value += 1
            if next_min_value > UPPER_LIMIT:
                next_min = None
        if next_min is None:
            arr = update_arr(arr)
        else:
            arr.append(next_min_value)
            if next_min[1]:
                total += 1
                print ('Found {}, Double check sum {}, total = {}'.format(arr, inverse_square_sum2(arr), total))
    print ('total is {}'.format(total))


find_sequence()








def lcm(x, y):
    return (x*y)//math.gcd(x,y)

assert lcm(2,4) == 4
assert lcm(6,9) == 18
assert lcm(5,7) == 35
assert lcm(5460, 1260) == 16380

def multiple_lcm(arr):
    l = arr[0]
    for i in range(1, len(arr)):
        l = lcm(l, arr[i])
    return l

assert multiple_lcm([2,3]) == 6
assert multiple_lcm([2,3,4]) == 12
assert multiple_lcm([2,3,4,5,7,12,15,20,28,35]) == 420
assert multiple_lcm([2,3,4,6,7,9,10,20,28,35,36,45]) == 1260
assert multiple_lcm([2,3,4,6,7,9,12,15,28,30,35,36,45]) == 1260
assert multiple_lcm([2, 3, 4, 5, 7, 12, 13, 28, 35, 39, 52]) == 5460
assert multiple_lcm([2, 3, 4, 5, 7, 12, 15, 21, 28, 42, 60, 70]) == 420



solutions = [
[2, 3, 4, 5, 7, 10, 20, 28, 30, 35, 60],
[2, 3, 4, 5, 7, 12, 13, 28, 35, 39, 52],
[2, 3, 4, 5, 7, 12, 15, 20, 28, 35],
[2, 3, 4, 5, 7, 12, 15, 21, 28, 42, 60, 70],
[2, 3, 4, 5, 7, 13, 15, 20, 28, 35, 39, 52],
[2, 3, 4, 5, 7, 13, 15, 21, 28, 39, 42, 52, 60, 70],
[2, 3, 4, 5, 10, 12, 13, 14, 15, 28, 30, 39, 42, 52],
[2, 3, 4, 6, 7, 9,  10, 20, 28, 35, 36, 45],
[2, 3, 4, 6, 7, 9,  12, 15, 28, 30, 35, 36, 45],
[2, 3, 4, 6, 7, 10, 12, 13, 21, 28, 39, 42, 52, 70],
[2, 3, 4, 6, 7, 10, 12, 14, 20, 21, 30, 60],
[2, 3, 4, 6, 7, 10, 13, 14, 20, 21, 30, 39, 52, 60],
[2, 3, 4, 6, 7, 10, 13, 15, 20, 21, 28, 39, 42, 52, 70],
[2, 3, 4, 6, 7, 10, 13, 15, 20, 21, 28, 39, 42, 52, 70],
[2, 3, 4, 6, 7, 12, 13, 14, 15, 20, 21, 39, 52]]

for s in solutions:
    print(inverse_square_sum2(s), multiple_lcm(s))

print('time spent is {}'.format(time.time() - now))

