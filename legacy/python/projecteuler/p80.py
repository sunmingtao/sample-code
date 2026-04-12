import math

limit = 100

def divide(m, n):
    count = 0
    base = m // n
    answer = str(base)
    count += len(answer)
    remainder = m % n * 10
    while count < limit and remainder != 0:
        new_base = remainder // n
        remainder = remainder % n * 10
        answer += str(new_base)
        count += 1
    return answer

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

def find_ab(a, b, c):
    return a * b + c, b

def find_seqs(n):
    block = find_block(n)
    seqs = [block[0]]
    block_seq = block[1]
    block_seq_len = len(block_seq)
    for i in range(seq_limit):
        seqs.append(block_seq[i % block_seq_len][0])
    return seqs

def find_bc(seqs):
    i = len(seqs) - 1
    b = seqs[i]
    c = 1
    a = seqs[i-1]
    while i >= 1:
        b, c = find_ab(a, b, c)
        i -= 1
        a = seqs[i-1]
    return b,c

seq_limit = 1000





len(divide(b,c))
sum(int(i) for i in divide(b,c))

total = 0
for i in range(2, 101):
    if int(math.sqrt(i)) ** 2 != i:
        b,c = find_bc(find_seqs(i))
        total += sum(int(j) for j in divide(b,c))

print(total)
b,c = find_bc(find_seqs(9))
print(i, divide(b,c))