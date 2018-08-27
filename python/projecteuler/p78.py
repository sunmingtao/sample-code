one_mil = 1000000

def p(m, n):
    if m <= 1 or n == 1:
        return 1
    else:
        if n > m:
            n = m
        sum = 0
        new_m = m - n
        new_n = n
        while new_n >= 1:
            sum += p(new_m, new_n)
            new_m += 1
            new_n -= 1
        return sum



def p2(m, n):
    if n > m:
        n = m
    if m <= 1 or n <= 1:
        return 1
    else:
        return get_arr_value(m, n-1) + get_arr_value(m-n, n)

def get_arr_value(m, n):
    if n > m:
        n = m
    if m <= 1 or n <= 1:
        return 1
    else:
        return p_arr[m][n] % one_mil


num = 80000

p_arr = [[-1 for x in range(num+1)] for y in range(num+1)]

for m in range(1, num+1):
    for n in range(1, m+1):
        p_arr[m][n] = p2(m, n)

p_arr[num][num]

p_arr_2 = [[0 for x in range(num+1)] for y in range(num+1)]
for m in range(1, num+1):
    for n in range(1, num+1):
        if n - 1 >= 0:
            p_arr_2[m][n] = p_arr[m][n] - p_arr[m][n-1]
        else:
            p_arr_2[m][n] = p_arr[m][n]

p_arr_2

for i in range(num+1):
    if p_arr[i][i] % 100000 == 0:
        print (i, p_arr[i][i])

for i in range(num+1):
    print(p_arr[i][i])

sum(p_arr[i][i] for i in range(11))
