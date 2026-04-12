import math

def power_list(n, max_num):
    result = []
    for i in range(1, max_num+1):
        result.append(i ** n)
    return result

power_list(20, 100)

result = []
for i in range(2, 21):
    result.extend(power_list(i, 100))

result = list(set(result))
result.sort()

print(result)

def good_num(num):
    digit_sum = sum(int(i) for i in str(num))
    power = 2
    while True:
        if digit_sum == 1:
            break
        else:
            digit_sum_power = digit_sum ** power
            if digit_sum_power == num:
                return True
            elif digit_sum_power > num:
                break
            else:
                power += 1
    return False


assert good_num(512)
assert good_num(614656)
assert not good_num(16)
assert not good_num(100)

index = 0
for i in result:
    if good_num(i):
        index += 1
        print (index, i)


