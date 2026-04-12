import math

def get_start_end(n):
    return n//3 +1, (n+1) // 2

total = 0
for n in range(2, 12001):
    start, end = get_start_end(n)
    seq = list(range(start, end))
    for i in seq:
        if math.gcd(i, n) == 1:
            total += 1

print(total)
