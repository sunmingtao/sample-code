import math

def get_all_primes(n):
    arr = [True for n in range(n+1)]
    arr[0], arr[1] = False, False
    arr[2] = True
    index = 2
    while index < len(arr):
        for j in range(index * 2, n+1, index):
            arr[j] = False
        index += 1
        while index < len(arr) and not arr[index]:
            index += 1
    return [i for i in range(n+1) if arr[i]]

all_primes = get_all_primes(100000000)
len(all_primes)


def has_duplicate_digit(num):
    return len(str(num)) != len(set(str(num)))

def has_zero(num):
    return "0" in str(num)

assert has_zero(10) == True
assert has_zero(1) == False

non_duplicate_primes = []

for i in all_primes:
    if not has_duplicate_digit(i) and not has_zero(i):
        non_duplicate_primes.append(i)


print(len(non_duplicate_primes))

print (non_duplicate_primes[25:107])


def is_prime_set(num_set):
    num_list_str = ''.join([str(i) for i in num_set])
    if not len(num_list_str) == 9:
        return False
    if not set('123456789') == set(num_list_str):
        return False
    return True

assert is_prime_set([2,3,5,7,89,461])
assert is_prime_set([2,3,5,7,89,641])
assert not is_prime_set([2,3,5,7,897,641])
assert not is_prime_set([2,3,5,7,89])
assert is_prime_set([2,5,47,89,631])


def prime_list_status(num_list, num_left):
    num_list_str = ''.join([str(i) for i in num_list])
    if len(num_list_str) + len(str(num_list[-1])) * num_left > 9:
        return 'over'
    elif has_duplicate_digit(num_list_str):
        return 'duplicate'
    else:
        return 'good'

assert prime_list_status([2,3,5], 3) == 'good'
assert prime_list_status([2,3,5,7,11], 1) == 'duplicate'
assert prime_list_status([2,5,47,89,146,631], 3) == 'over'
assert prime_list_status([3,5,11], 3) == 'over'

def convert_prime_index_to_prime(candidates):
    return [non_duplicate_primes[i] for i in candidates]

assert convert_prime_index_to_prime([0,1,2]) == [2,3,5]


def search_prime(result, candidates, num_left):
    if num_left == 0:
        if is_prime_set(convert_prime_index_to_prime(candidates)):
            result.append(candidates)
    else:
        prime_index = candidates[-1] + 1
        while prime_index < len(non_duplicate_primes):
            candidates_copy = candidates.copy()
            candidates_copy.append(prime_index)
            prime_candidates = convert_prime_index_to_prime(candidates_copy)
            status = prime_list_status(prime_candidates, num_left-1)
            if status == 'over':
                break
            elif status == 'good':
                search_prime(result, candidates_copy, num_left-1)
            prime_index += 1

result = []
search_prime(result, [0], 5)
assert len(result) == 5


def search(n):
    result = []
    for i in range(0, len(non_duplicate_primes) // 2):
        candidates = [i]
        prime_candidates = convert_prime_index_to_prime(candidates)
        status = prime_list_status(prime_candidates, n - 1)
        if status == 'over':
            break
        else:
            search_prime(result, candidates, n-1)
    return result


print (sum(len(search(i)) for i in range(2, 7)))
