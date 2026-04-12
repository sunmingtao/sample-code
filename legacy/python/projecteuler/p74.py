import math

def chain_num(n):
    seq = [n]
    new_n = sum_factorial_digit(n)
    while new_n not in seq:
        seq.append(new_n)
        new_n = sum_factorial_digit(new_n)
    return len(seq)

def sum_factorial_digit(n):
    return sum(math.factorial(int(i)) for i in str(n))


limit = 1000000

total=0
for i in range(limit):
    if chain_num(i) == 60:
        total += 1
        print(i)

print(total)