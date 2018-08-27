def digit_sum(a, b):
    return int(sum(int(i) for i in str(a ** b)))

digit_sum(3,3)

max_digit_sum = 0
for i in range(1, 100):
    for j in range(1, 100):
        digit_sum_ = digit_sum(i, j)
        if digit_sum_ > max_digit_sum:
            max_digit_sum = digit_sum_

max_digit_sum
