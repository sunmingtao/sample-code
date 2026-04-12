threshold = 1000000


def num_exceed(n):
    under = 1
    above = n
    i = 1
    while i <= n//2 and above // under <= threshold:
        above *= n - i
        under *= i + 1
        i += 1
    if above // under > threshold:
        result = (n // 2 + 1 - i) * 2
        if n % 2 == 0:
            result -= 1
        return result
    else:
        return 0


num_exceed(24)

sum(num_exceed(i) for i in range(1, 101))

for i in range(1, 101):
    print(i, num_exceed(i))
