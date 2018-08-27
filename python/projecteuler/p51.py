import math
import itertools

upper_bound = 1000000

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True

def has_duplicate_digit(num):
    return len(str(num)) != len(set(str(num)))

def min_duplicate_digits(num):
    count_dict = {}
    num_str = str(num)
    for i in range(len(num_str)):
        if num_str[i] in count_dict:
            count_dict[num_str[i]] += 1
        else:
            count_dict[num_str[i]] = 1
    duplicate_digits = [i for i in count_dict.keys() if count_dict[i] > 1]
    if len(duplicate_digits) == 0:
        return -1
    else:
        return int(min(duplicate_digits))



all_primes = [i for i in range(2, upper_bound) if is_prime(i) and has_duplicate_digit(i)]

all_5_digit_primes = [i for i in all_primes if i > 10000 and i < 99999]
all_5_digit_primes_min = [i for i in all_5_digit_primes if min_duplicate_digits(i) < 3]
all_6_digit_primes = [i for i in all_primes if i > 100000 and i < 999999]
all_6_digit_primes_min = [i for i in all_6_digit_primes if min_duplicate_digits(i) < 3]


len(all_6_digit_primes_min)



def has_same_digit_on_index(num, indexes):
    num_str = str(num)
    digit_at_indexes = [num_str[i] for i in indexes]
    return len(set(digit_at_indexes)) == 1

has_same_digit_on_index(10007, [1,2])

def replace_indexes(num):
    num_str = str(num)


def non_replace_index(length, replace_indexes):
    l = list(range(length))
    for i in replace_indexes:
        l.remove(i)
    return l

def join_by_index(s, indexes):
    return ''.join([s[i] for i in indexes])

join_by_index('12345', [2,3])

def get_replace_indexes(num):
    num_str = str(num)


for i in all_5_digit_primes:
    print (i)


all_primes = [i for i in range(2, upper_bound) if is_prime(i) and has_duplicate_digit(i)]

all_5_digit_primes = [i for i in all_primes if i > 10000 and i < 99999]
all_6_digit_primes = [i for i in all_primes if i > 100000 and i < 999999]


def same_family(num1, num2):
    num1_str = str(num1)
    num2_str = str(num2)
    not_same_indexes = []
    not_same_index_value1 = -1
    not_same_index_value2 = -1
    for i in range(len(num1_str)):
        if num1_str[i] != num2_str[i]:
            not_same_indexes.append(i)
            if not_same_index_value1 == -1:
                not_same_index_value1 = num1_str[i]
            elif not_same_index_value1 != num1_str[i]:
                return 0, tuple()
            if not_same_index_value2 == -1:
                not_same_index_value2 = num2_str[i]
            elif not_same_index_value2 != num2_str[i]:
                return 0, tuple()
    if len(not_same_indexes) > 1:
        return num1, tuple(not_same_indexes)
    else:
        return 0, tuple()

same_family(12322, 18388)

family_dict = {}
for num_i in all_6_digit_primes_min:
    for num_j in all_6_digit_primes:
        if (num_j > num_i):
            same_family_ = same_family(num_i, num_j)
            if same_family_[0] > 0:
                if same_family_ in family_dict:
                    family_dict.get(same_family_).append(num_j)
                else:
                    family_dict[same_family_]=[num_i, num_j]


test_dict = {'a':[1,2], 'b':[4,5], (1,2):[6,7]}
test_dict.get((1,2))

test_dict[(3,4)]=[8,9]
test_dict[(3,4)].append(10)

family_dict.get((56003, (2,3)))
[i for i in family_dict.values() if len(i) == 8]

for i in all_6_digit_primes:
    for j in all_6_digit_primes:
        i* j

len(all_6_digit_primes)

import bisect

all_primes[bisect.bisect(all_primes,5)-1]



min_duplicate_digits(12399)

def prob51():
    # test primes in list of primes under 10Meg
    for p in all_primes:
        if p < 10000:
            continue
        s = str(p)
        for c in '012':
            if s.count(c) <= 1:
                continue
            losers = []
            winners = []
            for d in '0123456789':
                n = int(s.replace(c,d))
                # see if candidate family member is prime
                if d >= c and n in all_primes:
                    winners.append(n)
                else:
                    losers.append(n)
            # if more than 2 losers, then can't be a
            # family of eight primes
            if len(losers) > 2:
                break
            print(winners)



prob51()


