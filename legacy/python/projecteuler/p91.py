import math

def min_divide(a , b):
    gcd = math.gcd(a, b)
    return int(a/gcd), int(b/gcd)

max = 50

max_list = list(range(max+1))

total = 0
for i in range(max+1):
    for j in range(max+1):
        if i == 0 and j == 0:
            continue
        total += count_right_angle(i, j)

print(total+max**2)


def count_right_angle(i, j):
    count = 0
    min_pair = min_divide(i, j)
    i0 = min_pair[0]
    j0 = min_pair[1]
    new_i_1 = i + j0
    new_j_1 = j - i0
    while new_i_1 in max_list and new_j_1 in max_list:
        count += 1
        new_i_1 += j0
        new_j_1 -= i0
    new_i_2 = i - j0
    new_j_2 = j + i0
    while new_i_2 in max_list and new_j_2 in max_list:
        count += 1
        new_i_2 -= j0
        new_j_2 += i0
    return count

print(count_right_angle(0,0))

min_divide(2,4)
