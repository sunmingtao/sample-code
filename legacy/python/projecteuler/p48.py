

len(str(999**999))

digit = 10

def pow(n):
    total = 1
    for i in range(n):
        total *= n
        total = int(str(total)[-digit:])
    return total

sum(pow(i) for i in range(1, 1001))


'1234567890'[-10:]