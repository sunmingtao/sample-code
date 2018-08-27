def find_ab(a, b, c):
    return a * b + c, b

a, b = find_ab(2,1,1)
a, b = find_ab(1, a, b)
find_ab(2, a, b)


def find_seqs():
    seqs = [2]
    for i in range(33):
        seqs.append(1)
        seqs.append((i+1)* 2)
        seqs.append(1)
    return seqs

seqs = find_seqs()

seqs = [3,1,1,1,1,6]
seqs = [7, 1, 4, 3, 1, 2, 2, 1, 3, 4, 1, 14]

i = len(seqs) - 1
b = seqs[i]
c = 1
a = seqs[i-1]
while i >= 1:
    b, c = find_ab(a, b, c)
    i -= 1
    a = seqs[i-1]

print(b,c)

import math
math.gcd(b,c)
sum(int(i) for i in str(b))








