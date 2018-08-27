import math

def find_base(n):
    sqrt = int(math.sqrt(n))
    if sqrt ** 2 - n == 0:
        return 0
    else:
        return sqrt

def find_abc(b,c, n):
    c_temp = (n - b ** 2) // c
    i = int(math.sqrt(n))
    while i>=1:
        if (i+b) % c_temp == 0:
            a_temp = (i+b) // c_temp
            b_temp = a_temp * c_temp - b
            break
        else:
            i -= 1
    return a_temp, b_temp, c_temp

def find_block(n):
    sequences = []
    base = find_base(n)
    a, b, c = find_abc(base, 1, n)
    sequences.append((a,b,c))
    next_sequence = find_abc(b, c, n)
    while next_sequence not in sequences:
        sequences.append(next_sequence)
        next_sequence = find_abc(next_sequence[1], next_sequence[2], n)
    return base, sequences

total = 0
for i in range(10001):
    if find_base(i) != 0:
        block = find_block(i)
        if len(block[1]) % 2 ==1:
            total += 1

total
find_abc(4, 1, 23)

find_block(61)



for i in range(100):
    print(i, find_base(i))

