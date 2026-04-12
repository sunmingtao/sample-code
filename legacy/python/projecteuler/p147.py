import math
import gmpy2
import time
import sympy

now = time.time()

def get_diagnal_length_list(m, n): # m >= n
    diagnal_length_list = []
    temp_list = []
    for i in range(1, n):
        diagnal_length_list.append(i * 2)
        temp_list.append(i * 2)
    for i in range(m - n):
        diagnal_length_list.append(2 * n - 1)
    diagnal_length_list.extend(temp_list[::-1])
    return diagnal_length_list

assert get_diagnal_length_list(3, 3) == [2, 4, 4, 2]
assert get_diagnal_length_list(6, 4) == [2, 4, 6, 7, 7, 6, 4, 2]


def get_part_diagnal_length_list(diagnal_length_list, start_index, width):
    end_index = min(start_index + width, len(diagnal_length_list))
    return diagnal_length_list[start_index:end_index]

assert get_part_diagnal_length_list([2, 4, 6, 7, 7, 6, 4, 2], 0, 3) == [2, 4, 6]
assert get_part_diagnal_length_list([2, 4, 6, 7, 7, 6, 4, 2], 0, 1) == [2]
assert get_part_diagnal_length_list([2, 4, 6, 7, 7, 6, 4, 2], 0, 10) == [2, 4, 6, 7, 7, 6, 4, 2]


def get_n(diagnal_length_list):
    max_item = max(diagnal_length_list)
    if diagnal_length_list[-1] == max_item and diagnal_length_list.count(max_item) == 1:
        return 0
    else:
        if max_item % 2 == 1:
            return (max_item + 1) // 2
        else:
            return max_item // 2 + 1


assert get_n([2]) == 0
assert get_n([2, 4]) == 0
assert get_n([2, 4, 4]) == 3
assert get_n([2, 4, 4, 2]) == 3
assert get_n([2, 4, 5, 4, 2]) == 3
assert get_n([2, 4, 5, 5, 5]) == 3


def get_max_length(diagnal_length_list):
    n = get_n(diagnal_length_list)
    if n == 0:
        return min(diagnal_length_list)
    elif diagnal_length_list[-1] < diagnal_length_list[0]:
        return get_max_length(diagnal_length_list[::-1])
    else:
        remaining = n * 2 - diagnal_length_list[0]
        if len(diagnal_length_list) <= remaining:
            return diagnal_length_list[0]
        else:
            return diagnal_length_list[0] - (len(diagnal_length_list) - remaining)


assert get_max_length([2]) == 2
assert get_max_length([6]) == 6
assert get_max_length([4, 4]) == 4
assert get_max_length([6, 6]) == 6
assert get_max_length([4, 6, 6]) == 4
assert get_max_length([4, 6, 8, 8, 6]) == 4
assert get_max_length([6, 6, 4]) == 4
assert get_max_length([4, 5, 5]) == 3
assert get_max_length([4, 6, 7]) == 4
assert get_max_length([6, 7, 7]) == 5
assert get_max_length([6, 7, 7, 6]) == 4
assert get_max_length([6, 8, 9, 9]) == 6
assert get_max_length([6, 8, 8]) == 6
assert get_max_length([6, 8, 8, 6]) == 6
assert get_max_length([8, 10, 10, 8]) == 8
assert get_max_length([7, 7, 6]) == 5
assert get_max_length([7, 7, 7]) == 5
assert get_max_length([7, 7, 7, 7]) == 4
assert get_max_length([7, 7, 6, 4]) == 4
assert get_max_length([6, 4]) == 4
assert get_max_length([6, 4, 2]) == 2
assert get_max_length([4, 2]) == 2
assert get_max_length([9, 9, 9, 9]) == 6
assert get_max_length([9, 9, 9, 9, 9]) == 5
assert get_max_length([14, 16, 18, 19, 19, 19, 19, 19, 19]) == 11
assert get_max_length([19, 19, 19, 19, 19, 19, 18, 16, 14]) == 11
temp = [24, 26, 28, 30, 32, 34, 36, 38, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18]
assert get_max_length(temp) == get_max_length(temp[::-1])



def calculate_non_square(diagnal_length_list):
    total_non_square_rectangle = 0
    for i in range(len(diagnal_length_list)):
        length = diagnal_length_list[i]
        temp_total = 0
        for width in range(1, length):
            part_diagnal_length_list = get_part_diagnal_length_list(diagnal_length_list, i, width)
            max_length = get_max_length(part_diagnal_length_list)
            if max_length != get_max_length(part_diagnal_length_list[::-1]):
                print ('error', part_diagnal_length_list, max_length, get_max_length(part_diagnal_length_list[::-1]))
            if max_length <= width:
                break
            total_non_square_rectangle += (1 + max_length - width) * (max_length - width) // 2
            temp_total += (1 + max_length - width) * (max_length - width) // 2
            #print (part_diagnal_length_list, max_length, (1 + max_length - width) * (max_length - width) // 2)
    return total_non_square_rectangle * 2


assert(calculate_non_square(get_diagnal_length_list(2, 2))) == 4
assert(calculate_non_square(get_diagnal_length_list(3, 1))) == 0
assert(calculate_non_square(get_diagnal_length_list(3, 2))) == 10
assert(calculate_non_square(get_diagnal_length_list(3, 3))) == 34
assert(calculate_non_square(get_diagnal_length_list(4, 3))) == 60
assert(calculate_non_square(get_diagnal_length_list(4, 4))) == 124
assert(calculate_non_square(get_diagnal_length_list(5, 4))) == 192
assert(calculate_non_square(get_diagnal_length_list(6, 4))) == 260



def calculate_square(m, n):
    if m < n:
        raise ValueError("m cannot be less than n")
    total_square = 0
    for i in range(1, n+1):
        if i % 2 == 1:
            total_square += (m - i) * (n - i + 1) + (m - i + 1) * (n - i)
        else:
            total_square += (m - i + 1) * (n - i + 1) + (m - i) * (n - i)
    return total_square

assert calculate_square(1, 1) == 0
assert calculate_square(2, 1) == 1
assert calculate_square(2, 2) == 5
assert calculate_square(3, 3) == 17


def calculate_diagnal_rectangle(m, n):
    return calculate_non_square(get_diagnal_length_list(m, n)) + calculate_square(m, n)

assert calculate_diagnal_rectangle(3, 2) == 19

def calculate_normal_rectangle(m, n):
    return m * (m + 1) * n * (n + 1) // 4

assert calculate_normal_rectangle(1, 1) == 1
assert calculate_normal_rectangle(2, 2) == 9
assert calculate_normal_rectangle(2, 1) == 3
assert calculate_normal_rectangle(3, 2) == 18

def calculate_all_rectangle(m, n):
    return calculate_diagnal_rectangle(m, n) + calculate_normal_rectangle(m, n)

assert calculate_all_rectangle(3, 2) == 37
assert calculate_all_rectangle(1, 1) == 1
assert calculate_all_rectangle(2, 1) == 4
assert calculate_all_rectangle(3, 1) == 8
assert calculate_all_rectangle(2, 2) == 18

def calculate_within_all_rectangle(m, n):
    total = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if i >= j:
                rectangle_num = calculate_all_rectangle(i, j)
            else:
                rectangle_num = calculate_all_rectangle(j, i)
            total += rectangle_num
    return total


assert calculate_within_all_rectangle(3, 2) == 72
print (calculate_within_all_rectangle(47, 43))

print('time spent is {}'.format(time.time() - now))