import math


upper_bound = 10000

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True

def same_digit(num1, num2):
    str_num1 = str(num1)
    str_num2 = str(num2)
    for i in str_num1:
        if str_num1.count(i) != str_num2.count(i):
            return False
    return True

all_primes = [i for i in range(1000, upper_bound) if is_prime(i)]

for i in all_primes:
    for j in all_primes:
        if j > i and same_digit(i, j):
            k = j + j - i
            if k in all_primes and same_digit(i, k):
                print(i,j,k)

len(all_primes)

