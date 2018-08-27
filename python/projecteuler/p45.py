from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1


def t(n):
    return int(n * (n + 1) / 2)

def p(n):
    return int(n * (3 * n - 1) / 2)

def h(n):
    return int(n * (2 * n - 1))

max_n = 100000

t_list = [t(n) for n in range(1, max_n+1)]
p_list = [p(n) for n in range(1, max_n+1)]
h_list = [h(n) for n in range(1, max_n+1)]

for n in t_list:
    if index(p_list, n) != -1 and index(h_list, n) != -1:
        print(n)
