import math
import gmpy2
import time
import sympy
import numpy as np

now = time.time()

def init_arr():
    arr = []
    t = 0
    for k in range(500500):
        t = (615949 * t + 797807) % 2 ** 20
        arr.append(t - 2 ** 19)
    return arr

arr = init_arr()
assert arr[0] == 273519
assert arr[1] == -153582
assert arr[2] == 450905


def find_smallest_triangle(arr, max_depth):
    total_map = {index: arr[index] for index in range(len(arr))}
    min_map = {index: arr[index] for index in range(len(arr))}
    for d in range(max_depth):
        for i in range(d):
            arr_i = d * (d + 1) // 2 + i
            sub_total = arr[arr_i]
            temp_arr_index = arr_i
            for j in range(i+1, d+1):
                arr_j = d * (d + 1) // 2 + j
                sub_total += arr[arr_j]
                temp_arr_index -= d - (j - i - 1)
                total_map[temp_arr_index] += sub_total
                if total_map[temp_arr_index] < min_map[temp_arr_index]:
                    min_map[temp_arr_index] = total_map[temp_arr_index]
    return min(total for _, total in min_map.items())


assert find_smallest_triangle([15, -14, -7, 20, -13, -5, -3, 8, 23, -26, 1, -4, -5, -18, 5, -16, 31, 2, 9, 28, 3], 6) == -42

print(find_smallest_triangle(arr, 1000))



print('time spent is {}'.format(time.time() - now))
