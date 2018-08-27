upper_bound = 10000

def reverse(num):
    num_str = str(num)
    return int(num_str[::-1])

reverse(120)

def is_palindrome(num):
    return num == reverse(num)

for i in range(50)

def is_lychrel(num):
    i = 1
    new_num = num + reverse(num)
    while i <= 50 and not is_palindrome(new_num):
        new_num = new_num + reverse(new_num)
        i += 1
    if is_palindrome(new_num):
        return False
    else:
        return True

sum(is_lychrel(num) for num in range(1, upper_bound))

