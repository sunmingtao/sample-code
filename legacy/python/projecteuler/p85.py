def count_blocks(m,n):
    total = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            total += (m+1-i)*(n+1-j)
    return total

def count_blocks2(m, n):
    return (m ** 2 + m) * (n ** 2 + n) // 4


two_mil = 2000000
closest = 2000000
good_m, good_n = 0, 0
for m in range(1, 100):
    for n in range(1, 100):
        blocks = count_blocks2(m, n)
        diff = two_mil - blocks
        if 0 < diff < closest:
            closest = diff
            good_m = m
            good_n = n

print(good_m, good_n, good_m * good_n)

count_blocks2(36, 77)