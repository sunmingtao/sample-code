import math

upper_bound = 1000000

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True

def has_no_leading_even_digit(num):
    return any([int(i) % 2 == 0 for i in str(num)[1:]])

def has_no_leading_5_digit(num):
    return any([int(i) % 5 == 0 for i in str(num)[1:]])

all_primes = [i for i in range(2, upper_bound) if is_prime(i) and not has_no_leading_even_digit(i)
              and not has_no_leading_5_digit(i)]

def truncate_left(s):
    return s[1:]

def truncate_right(s):
    return s[:-1]

def all_truncate_left(num):
    num_str = str(num)
    result = []
    for i in range(1, len(num_str)):
        left_trun = truncate_left(num_str)
        num_str = left_trun
        result.append(left_trun)
    return result

def all_truncate_right(num):
    num_str = str(num)
    result = []
    for i in range(1, len(num_str)):
        right_trun = truncate_right(num_str)
        num_str = right_trun
        result.append(right_trun)
    return result



def has_all(trun):
    return all([int(i) in all_primes for i in trun])

count=0;
for a in all_primes:
    all_trun = all_truncate_left(a)
    all_trun.extend(all_truncate_right(a))
    if a> 10 and has_all(all_trun):
        print(a)
        count+=a

count

