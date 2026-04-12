import math
import gmpy2
import time
import sympy

now = time.time()

def get_next_row(current_row):
    next_row = [1]
    for i in range(len(current_row)-1):
        next_row.append((current_row[i]+current_row[i+1]) % 7)
    next_row.append(1)
    return next_row


assert get_next_row([1]) == [1, 1]
assert get_next_row([1, 1]) == [1, 2, 1]
assert get_next_row([1, 2, 1]) == [1, 3, 3, 1]
assert get_next_row([1, 5, 10, 10, 5, 1]) == [1, 6, 1, 6, 1, 6, 1]
assert get_next_row([1, 6, 1, 6, 1, 6, 1]) == [1, 0, 0, 0, 0, 0, 0, 1]


def count_divisible_num(row):
    return sum(i % 7 == 0 for i in row)

assert count_divisible_num([1]) == 0
assert count_divisible_num([1, 7, 21, 35, 35, 21, 7, 1]) == 6
assert count_divisible_num([1, 0, 0, 0, 0, 0, 0, 1]) == 6



def count7(num):
    current_row = [1]
    divisible_num = 0
    i = 1
    while i < num:
        next_row = get_next_row(current_row)
        divisible_num += count_divisible_num(next_row)
        current_row = next_row
        i+=1
    return (num+1)*num // 2 - divisible_num

assert count7(1) == 1
assert count7(2) == 3
assert count7(7) == 28 # 7 ** 1
assert count7(49) == 784 # 7 ** 2
assert count7(7 ** 2 + 1) == count7(49) + count7(2) - count7(1)  # 786
assert count7(343) == 21952 # 7 ** 3
assert count7(343 * 2) == 65856 #  * 3
assert count7(343 * 3) == 131712 #  * 6
assert count7(343 * 4) == 219520 #  * 10
assert count7(343 * 5) == 329280 #  * 15
assert count7(343 * 6) == 460992 #  * 21
assert count7(343 * 6 + 1) == count7(343 * 6) + 7
assert count7(343 * 6 + 2) == count7(343 * 6) + 7 + 14
assert count7(343 * 6 + 7 * 1) == count7(343 * 6) + count7(7 * 7) - count7(7 * 6) # 461188
assert count7(343 * 6 + 7 * 1 + 1) == count7(343 * 6) + count7(7 * 7) - count7(7 * 6) + 14 # 461202
assert count7(343 * 6 + 49 * 1) == count7(343 * 6) + 7 * count7(49) # 466480
assert count7(343 * 6 + 49 * 1 + 1) == count7(343 * 6) + 7 * count7(49) + 14 #466494
assert count7(343 * 6 + 49 * 1 + 2) == count7(343 * 6) + 7 * count7(49) + 42 #466522
assert count7(343 * 6 + 49 * 2) == count7(343 * 6) + 7 * count7(49 * 2)
assert count7(343 * 6 + 49 * 2 + 1) == count7(343 * 6) + 7 * count7(49 * 2) + 21
assert count7(343 * 5 + 49 * 4) == count7(343 * 5) + 6 * count7(49 * 4)
assert count7(343 * 5 + 49 * 4 + 7 * 3 + 2) == count7(343 * 5) + (5 + 1) * count7(49 * 4) + (5 + 1)*(4 + 1) * count7(7 * 3) + (5+1)*(4+1)*(3+1) * count7(2)
assert count7(343 * 5 + 7 * 3 + 2) == count7(343 * 5) + (5 + 1) * count7(7 * 3) + (5+1)*(3+1) * count7(2)

assert count7(2401) == 614656 # 7 ** 4
assert count7(2401 * 2) == 1843968 # + 56


assert count7(100) == 2361
assert count7(1000) == 118335


def get_power_7(num):
    power = int(math.log(num, 7))
    multiple = num // 7 ** power
    return power, multiple, num - 7 ** power * multiple

assert get_power_7(10 ** 9) == (10, 3, 152574253)

def disassemble(num):
    result = []
    power, multiple, remainder = get_power_7(num)
    result.append((power, multiple))
    while remainder != 0:
        power, multiple, remainder = get_power_7(remainder)
        result.append((power, multiple))
    return result

assert disassemble(7) == [(1, 1)]
assert disassemble(8) == [(1, 1), (0, 1)]

assert sum(7 ** i * j for i, j in disassemble(10 ** 9)) == 10 ** 9

def count_by_power_multiple(power, multiple):
    return 28 ** power * (multiple + 1) * multiple // 2

assert count_by_power_multiple(3, 6) == 460992

def count7_v2(num):
    total = 0
    coefficient = 1
    for power, multiple in disassemble(num):
        total += coefficient * count_by_power_multiple(power, multiple)
        coefficient *= multiple + 1
    return total

assert count7_v2(343 * 5 + 49 * 4 + 7 * 3 + 2) == count7(343 * 5 + 49 * 4 + 7 * 3 + 2)

print(count7_v2(10 ** 9))

print('time spent is {}'.format(time.time() - now))

