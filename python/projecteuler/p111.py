import math
import itertools

def is_prime(num):
    for i in range(2, int(math.pow(num, 0.5))+1):
        if num % i == 0:
            return False
    return True

assert is_prime(1111111111) == 0

def all_digits_minus_one_digit(digit):
    return set('1234567890') - set(str(digit))

assert all_digits_minus_one_digit('1') == {'0', '2', '3', '4', '5', '6', '7', '8', '9'}


def get_last_index(_list, item):
    if item in _list:
        return len(_list) - _list[::-1].index(item) - 1
    else:
        return -1


assert get_last_index('1122', '1') == 1
assert get_last_index('1122', '2') == 3
assert get_last_index('1122', '3') == -1




def add_item_to_permutation_list(item, permutation_list):
    new_permutation_list = []
    if len(permutation_list) == 0:
        new_permutation_list.append([item])
    else:
        for permutation in permutation_list:
            last_index = get_last_index(permutation, item)
            for index in range(last_index+1, len(permutation)+1):
                permutation_copy = permutation.copy()
                permutation_copy.insert(index, item)
                new_permutation_list.append(permutation_copy)
    return new_permutation_list


assert add_item_to_permutation_list('1', []) == [['1']]
assert add_item_to_permutation_list('1', [['1']]) == [['1','1']]
assert add_item_to_permutation_list('2', [['1']]) == [['2','1'],['1','2']]
assert add_item_to_permutation_list('1', [['1','1']]) == [['1','1', '1']]
assert add_item_to_permutation_list('2', [['1','1']]) == [['2', '1','1'], ['1', '2', '1'], ['1', '1', '2']]


def no_repeat_permutation(items):
    item_list = list(items)
    item_list.sort()
    permutation_list = []
    for item in item_list:
        permutation_list = add_item_to_permutation_list(item, permutation_list)
    return permutation_list


assert no_repeat_permutation('1') == [['1']]
assert no_repeat_permutation('11') == [['1','1']]
assert no_repeat_permutation('12') == [['2','1'], ['1', '2']]
assert no_repeat_permutation('112') == [['2','1','1'], ['1','2','1'], ['1','1','2']]
assert no_repeat_permutation('1122') == [['2', '2', '1','1'], ['2', '1', '2', '1'], ['2', '1', '1', '2'], ['1','2', '2', '1'], ['1','2','1', '2',], ['1','1','2','2']]
assert no_repeat_permutation('2121') == [['2', '2', '1','1'], ['2', '1', '2', '1'], ['2', '1', '1', '2'], ['1','2', '2', '1'], ['1','2','1', '2',], ['1','1','2','2']]
assert len(no_repeat_permutation('apple')) == 60


def get_repeat_digit_candidate_list(length, repeated_digits, test_length):
    assert test_length < length
    base_candidate_list = str(repeated_digits) * test_length
    minus_repeated_digits_set = all_digits_minus_one_digit(repeated_digits)
    non_repeated_digits_list = get_non_repeated_digits_list(minus_repeated_digits_set, length - test_length)
    candidate_list = []
    for non_repeated_digits in non_repeated_digits_list:
        candidate_list.append(base_candidate_list + ''.join(non_repeated_digits))
    return candidate_list


def add_to_non_repeated_digits_list(digits_list, pickup_list):
    new_pickup_list = []
    if len(pickup_list) == 0:
        for digit in digits_list:
            new_pickup_list.append([digit])
    else:
        for pickup in pickup_list:
            max_digit = max(pickup)
            for index in range(digits_list.index(max_digit), len(digits_list)):
                pickup_copy = pickup.copy()
                pickup_copy.append(digits_list[index])
                new_pickup_list.append(pickup_copy)
    return new_pickup_list


assert add_to_non_repeated_digits_list('123', []) == [['1'], ['2'], ['3']]
assert add_to_non_repeated_digits_list('123', [['1'], ['2'], ['3']]) == [['1', '1'], ['1', '2'], ['1', '3'], ['2', '2'], ['2', '3'], ['3', '3']]
assert add_to_non_repeated_digits_list('123', [['1', '1'], ['1', '2'], ['1', '3'], ['2', '2'], ['2', '3'], ['3', '3']]) == [['1', '1', '1'], ['1', '1', '2'], ['1', '1', '3'], ['1', '2', '2'], ['1', '2', '3'], ['1', '3', '3'], ['2', '2', '2'], ['2', '2', '3'], ['2', '3', '3'], ['3', '3', '3']]


def get_non_repeated_digits_list(digits_set, n):
    digits_list = list(digits_set)
    digits_list.sort()
    pickup_list = []
    for i in range(n):
        pickup_list = add_to_non_repeated_digits_list(digits_list, pickup_list)
    return pickup_list


assert get_non_repeated_digits_list('123', 1) == [['1'], ['2'], ['3']]
assert get_non_repeated_digits_list('123', 2) == [['1', '1'], ['1', '2'], ['1', '3'], ['2', '2'], ['2', '3'], ['3', '3']]
assert get_non_repeated_digits_list('123', 3) == [['1', '1', '1'], ['1', '1', '2'], ['1', '1', '3'], ['1', '2', '2'], ['1', '2', '3'], ['1', '3', '3'], ['2', '2', '2'], ['2', '2', '3'], ['2', '3', '3'], ['3', '3', '3']]



def mns(n, d):
    test_length = n - 1
    while test_length >= 1:
        prime_list = []
        base_candidate_list = get_repeat_digit_candidate_list(n, d, test_length)
        for base_candidate in base_candidate_list:
            permutated_candidate_list = no_repeat_permutation(base_candidate)
            for permutated_candidate in permutated_candidate_list:
                permutated_candidate = ''.join(permutated_candidate)
                if permutated_candidate[0] != '0':
                    if is_prime(int(permutated_candidate)):
                        prime_list.append(int(permutated_candidate))
        if len(prime_list) > 0:
            break
        else:
            test_length -= 1
    return test_length, len(prime_list), sum(prime_list)


assert mns(4,0) == (2, 13, 67061)
assert mns(4,1) == (3, 9, 22275)
assert mns(4,2) == (3, 1, 2221)
assert mns(4,3) == (3, 12, 46214)
assert mns(4,4) == (3, 2, 8888)
assert mns(4,5) == (3, 1, 5557)
assert mns(4,6) == (3, 1, 6661)
assert mns(4,7) == (3, 9, 57863)
assert mns(4,8) == (3, 1, 8887)
assert mns(4,9) == (3, 7, 48073)


def sum_s(n):
    total = 0
    for i in range(10):
        mm, nn, s = mns(n, i)
        print (i, mm, nn, s)
        total += s
    return total


assert sum_s(4) == 273700

print(sum_s(10))




