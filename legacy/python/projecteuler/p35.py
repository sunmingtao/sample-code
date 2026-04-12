import math


def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True


def has_even_digit(num):
    return any([int(i) % 2 == 0 for i in str(num)])

upper_bound = 1000000

all_primes = [i for i in range(2, upper_bound) if is_prime(i) and not has_even_digit(i)]
all_primes.insert(0, 2)


def has_all_rotations(rot):
    return all([int(i) in all_primes for i in rot])


def all_rotations(num):
    num_str = str(num)
    result = [num_str]
    for i in range(1, len(num_str)):
        right_rotated = right_rotation(num_str)
        num_str = right_rotated
        result.append(right_rotated)
    return result


def right_rotation(num_str):
    return num_str[-1] + num_str[0:-1]


count=0;
for a in all_primes:
    all_rot = all_rotations(a)
    if has_all_rotations(all_rot):
        count+=1

count

