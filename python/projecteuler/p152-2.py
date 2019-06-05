import math
import gmpy2
import time
import sympy
import numpy as np

now = time.time()

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


def greater_than_half(a):
    return a > 0.5 and (not is_half(a))

assert greater_than_half(0.5000055245087296)
assert not greater_than_half(1/2)
assert not greater_than_half(0.500000000000005)

CANDIDATES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 39, 40, 42, 45, 52, 56, 60, 63, 70, 72]
len(CANDIDATES)

def inverse_square_sum(arr):
    return sum(1 / i ** 2 for i in arr)


def find_prod(arr):
    p = 1
    for i in arr:
        p *= i ** 2
    return p

def inverse_square_sum2(arr):
    prod = find_prod(arr)
    return prod / sum(prod // i ** 2 for i in arr)


def inverse_square(num):
    return 1 / num ** 2

START_ARR = [2, 3]

def update_working_arr(arr, arr_inverse_square_sum):
    index = CANDIDATES.index(arr[-1])
    temp_arr = arr.copy()
    if index == len(CANDIDATES) - 1:
        del temp_arr[-1]
        old_temp_arr_value = temp_arr[-1]
        temp_arr[-1] = CANDIDATES[CANDIDATES.index(temp_arr[-1])+1]
        return update_working_arr(temp_arr, arr_inverse_square_sum - inverse_square(arr[-1]) - inverse_square(old_temp_arr_value) + inverse_square(temp_arr[-1]))
    else:
        temp_index = index+1
        temp_arr_inverse_square_sum = arr_inverse_square_sum + inverse_square(CANDIDATES[temp_index])
        while greater_than_half(temp_arr_inverse_square_sum) and temp_index < len(CANDIDATES) - 1:
            temp_index += 1
            temp_arr_inverse_square_sum = arr_inverse_square_sum + inverse_square(CANDIDATES[temp_index])
        if greater_than_half(temp_arr_inverse_square_sum) and temp_index == len(CANDIDATES) - 1:
            old_temp_arr_value = temp_arr[-1]
            temp_arr[-1] = CANDIDATES[CANDIDATES.index(temp_arr[-1]) + 1]
            return update_working_arr(temp_arr, arr_inverse_square_sum - inverse_square(old_temp_arr_value) + inverse_square(temp_arr[-1]))
        if is_half(temp_arr_inverse_square_sum):
            temp_arr.append(CANDIDATES[temp_index])
            return temp_arr, 0.5
        else:
            if less_than_half(arr_inverse_square_sum + inverse_square_sum(CANDIDATES[temp_index:])):
                if temp_index - index == 1:
                    del temp_arr[-1]
                    if len(temp_arr) == 0:
                        return [], 0
                    else:
                        old_temp_arr_value = temp_arr[-1]
                        temp_arr[-1] = CANDIDATES[CANDIDATES.index(temp_arr[-1]) + 1]
                        return update_working_arr(temp_arr, arr_inverse_square_sum - inverse_square(arr[-1]) - inverse_square(old_temp_arr_value) + inverse_square(temp_arr[-1]))
                else:
                    old_temp_arr_value = temp_arr[-1]
                    temp_arr[-1] = CANDIDATES[CANDIDATES.index(temp_arr[-1]) + 1]
                    return update_working_arr(temp_arr, arr_inverse_square_sum - inverse_square(old_temp_arr_value) + inverse_square(temp_arr[-1]))
            else:
                temp_arr.append(CANDIDATES[temp_index])
                return temp_arr, temp_arr_inverse_square_sum


assert update_working_arr([2], inverse_square_sum([2])) == ([2, 3], inverse_square_sum([2,3]))
arr = [2, 3, 4, 5, 6, 21, 24, 30, 35, 36, 40, 45, 56, 60, 72]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 5, 6, 21, 24, 30, 35, 36, 40, 45, 56, 63, 70], inverse_square_sum([2, 3, 4, 5, 6, 21, 24, 30, 35, 36, 40, 45, 56, 63, 70]))
arr = [2, 3, 4, 5, 6]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 5, 6, 12], inverse_square_sum([2, 3, 4, 5, 6, 12]))
arr = [2, 4]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([], 0)
arr = [2, 3, 4, 7, 9]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 7, 9, 10], inverse_square_sum([2, 3, 4, 7, 9, 10]))
arr = [2, 3, 4, 7, 8, 12]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 7, 9, 10], inverse_square_sum([2, 3, 4, 7, 9, 10]))
arr = [2, 3, 4, 5, 6, 21, 24, 30, 35, 36, 40, 45, 56, 60]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 5, 6, 21, 24, 30, 35, 36, 40, 45, 56, 60, 72], 0.5)
arr = [2, 3, 4, 5, 7, 9, 28, 35, 36, 39]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 5, 7, 9, 28, 35, 36, 42, 70], inverse_square_sum([2, 3, 4, 5, 7, 9, 28, 35, 36, 42, 70]))
arr = [2, 3, 4, 5, 7, 12, 18, 21, 28, 35, 36, 39]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 5, 7, 12, 18, 21, 28, 35, 36, 42, 63], 0.5)
arr = [2, 3, 4, 5, 7, 12, 18, 21, 28, 35, 36, 40]
assert update_working_arr(arr, inverse_square_sum(arr)) == ([2, 3, 4, 5, 7, 12, 18, 21, 28, 35, 36, 42, 63], 0.5)


def find_sequence():
    total = 0
    working_arr = START_ARR
    working_arr_list = update_working_arr(working_arr, inverse_square_sum(working_arr))
    while working_arr_list[1] > 0:
        #print (working_arr_list[0])
        if working_arr_list[1] == 0.5:
            total += 1
            print ('Found {}, double check {}, total = {}'.format(working_arr_list[0], inverse_square_sum2(working_arr_list[0]), total))
        working_arr_list = update_working_arr(working_arr_list[0], working_arr_list[1])

find_sequence()

print('time spent is {}'.format(time.time() - now))

