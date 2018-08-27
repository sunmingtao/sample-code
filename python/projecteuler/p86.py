import math

def find_abcd(m, n):
    a = m ** 2 - n ** 2
    b = 2 * m * n
    c = m ** 2 + n ** 2
    d = a+b+c
    return a,b,c,d

def ways(a,b):
    return ways_small(a,b)+ways_big(a,b)

def ways_big(a, b):
    small, big = find_small_big(a, b)
    count = 0
    temp_n = big - small
    while temp_n < small and temp_n <= big // 2:
        count += 1
        temp_n += 1
    return count

def ways_small(a, b):
    small, big = find_small_big(a, b)
    if big <= M_limit:
        return small // 2
    else:
        return 0

ways_small(20, 99)

def find_small_big(a, b):
    small, big = a, b
    if small > big:
        small, big = b, a
    return small, big


limit = 1000
M_limit = 1818

total = 0
for m in range(2, limit):
    for n in range(1, m):
        if ((m % 2 == 0 and n % 2 != 0) or (m % 2 != 0 and n % 2 == 0)) and math.gcd(m, n) == 1:
            a,b,c,d = find_abcd(m,n)
            small, big = find_small_big(a, b)
            temp_small, temp_big, temp_c = small, big, c
            while temp_small <= M_limit and temp_big <= 2 * M_limit:
                total += ways(temp_small, temp_big)
                #print(temp_small, temp_big)
                temp_small += small
                temp_big += big
                temp_c += c

print(total)



import math

def find_abc(a,b,c):
    lst = [a,b,c]
    lst.sort()
    return lst[0], lst[1], lst[2]

def is_square(n):
    sq = math.sqrt(n)
    return int(sq) ** 2 == n

is_square(10)

limit=24

total = 0
for i in range(1, limit+1):
    for j in range(1, i+1):
        for k in range(1, j+1):
            a, b, c= find_abc(i, j, k)
            if is_square((a+b) ** 2 + c** 2):
                if a+b == 24 or c == 24:
                    print(i,j,k, (a+b), c)
                total += 1

print(total)

