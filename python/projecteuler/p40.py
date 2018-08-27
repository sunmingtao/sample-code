'''
An irrational decimal fraction is created by concatenating the positive integers:

0.123456789101112131415161718192021...

It can be seen that the 12th digit of the fractional part is 1.

If dn represents the nth digit of the fractional part, find the value of the following expression.

d1 × d10 × d100 × d1000 × d10000 × d100000 × d1000000
'''

d1 = 1
123456789   9 * 1 = 9
101112....9899  90 * 2 = 180
100101102....998999    900 * 3 = 2700
9000 * 4 = 36000
90000 * 5 = 450000
900000 * 6 = 5400000

a = [9, 180, 2700, 36000, 450000, 5400000]
b = []
sum = 0
for i in a:
    sum += i
    b.append(sum)
b = [9, 189, 2889, 38889, 488889, 5888889]

[1, 10, 100, 1000, 10000, 100000, 1000000]

def d(n):
    num_digit, index = find_range(n)
    num_index = (index - 1) / num_digit
    num = 10 ** (num_digit - 1) + num_index
    return int(str(num)[(index - 1) % num_digit])

ssss = ''
for i in range(1, 190):
    ssss += str(d(i))
ssss

find_range(100)
prd = 1
for i in [1, 10, 100, 1000, 10000, 100000, 1000000]:
    prd *= d(i)
prd

def find_range(n):
    for i in range(len(b)):
        if n <= b[i]:
            if i - 1 < 0:
                return i+1, n
            else:
                return i+1, n - b[i-1]


