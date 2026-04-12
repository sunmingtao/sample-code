from bisect import bisect_left

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def square_plus(n):
    str_n = str(n)
    return sum(int(i) ** 2 for i in str_n)


def stuck(n):
    sq = square_plus(n)
    seq = [n, sq]
    while sq != 1 and sq != 89:
        sq = square_plus(sq)
        seq.append(sq)
    return sq, seq



sum(stuck(i)[0] == 89 for i in range(1,1000000))

lst = [i for i in range(1,1000)]

sum(list)

stuck(9999999)

s89 = []
s1 = []
for n in range(1, 600):
    if stuck(n)[0] == 89:
        s89.append(n)
    else:
        s1.append(n)

for n in range(600, 10000000):
    if index(s89, square_plus(n)) > -1:
        s89.append(n)
    else:
        s1.append(n)

print(len(s89))
