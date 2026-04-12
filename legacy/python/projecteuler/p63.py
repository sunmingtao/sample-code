def same_num_digit_power(n):
    i = 1
    num_digit = len(str(i ** n))
    count = 0
    while num_digit <= n:
        if num_digit == n:
            count += 1
        i += 1
        num_digit = len(str(i ** n))
    return count


limit = 10
sum(same_num_digit_power(n) for n in range(1,5000))