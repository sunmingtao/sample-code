def p(n):
    return int(n * (3 * n - 1) / 2)

from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1


upper_bound = 10000

p_list = [p(n) for n in range(1, upper_bound)]

for i in p_list:
    for j in p_list:
        if j > i:
            if index(p_list, j-i) != -1 and index(p_list, j+i) != -1:
                print (i, j, j-i)