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
    for i in range(limit):
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

def find_min_x(n):
    if int(math.sqrt(n)) ** 2 != n:
        seqs = find_seqs(n)
        for i in range(1, len(seqs)+1):
            seqs_sub = seqs[:i]
            b,c = find_bc(seqs_sub)
            if b ** 2 - c ** 2 * n == 1:
                return b,c,i
    return 0,0,0

limit = 100

find_min_x(4)

max_x = 0

for d in range(2,1001):
    b,c,i = find_min_x(d)
    if b > max_x:
        max_x = b
        print(d, b)

print(max_x)

seq = find_seqs(2)
b,c  = find_bc(seq)
print(b,c)





