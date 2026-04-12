def number_single_dart_by_multiple(n, multiple, highest_base):
    base = n / multiple
    if multiple == 3:
        candidate_list = find_triple(n)
    elif multiple == 2:
        candidate_list = find_double(n)
    elif multiple == 1:
        candidate_list = find_single(n)
    else:
        raise ValueError('Unknown multiple {}'.format(multiple))
    return base.is_integer() and base <= highest_base and base in candidate_list


assert number_single_dart_by_multiple(0, 3, 1) == 0
assert number_single_dart_by_multiple(1, 1, 1) == 1
assert number_single_dart_by_multiple(3, 3, 1) == 1
assert number_single_dart_by_multiple(4, 3, 1) == 0
assert number_single_dart_by_multiple(6, 3, 2) == 1
assert number_single_dart_by_multiple(6, 3, 1) == 0
assert number_single_dart_by_multiple(21, 1, 21) == 0
assert number_single_dart_by_multiple(25, 1, 25) == 1


def number_single_dart(n, highest_multiple, highest_base):
    if n == 0:
        return 1
    else:
        total = 0
        for i in range(1, highest_multiple+1):
            temp_highest_base = 25
            if i == highest_multiple:
                temp_highest_base = highest_base
            total += number_single_dart_by_multiple(n, i, temp_highest_base)
        return total


assert number_single_dart(0, 1, 1) == 1
assert number_single_dart(1, 1, 1) == 1
assert number_single_dart(1, 2, 1) == 1
assert number_single_dart(3, 3, 1) == 2
assert number_single_dart(6, 3, 3) == 3



def number_two_darts(n):
    if n == 0:
        return 1
    else:
        total = 0
        triples = find_triple(n)
        for i in triples:
            total += number_single_dart(n - i * 3, 3, i)
        doubles = find_double(n)
        for i in doubles:
            total += number_single_dart(n - i * 2, 2, i)
        singles = find_single(n)
        for i in singles:
            total += number_single_dart(n - i, 1, i)
        return total

assert number_two_darts(0) == 1
assert number_two_darts(1) == 1
assert number_two_darts(2) == 3
assert number_two_darts(3) == 4
assert number_two_darts(4) == 7


def number_checkout(n):
    doubles = find_double(n)
    total = 0
    for i in doubles:
        total += number_two_darts(n - 2 * i)
    return total

assert number_checkout(1) == 0
assert number_checkout(2) == 1
assert number_checkout(3) == 1
assert number_checkout(4) == 4
assert number_checkout(6) == 11


def find_triple(n):
    n = min(n, 60)
    return [i for i in range(1, n+1) if i * 3 <= n]

assert find_triple(0) == []
assert find_triple(1) == []
assert find_triple(2) == []
assert find_triple(3) == [1]
assert find_triple(4) == [1]
assert find_triple(4) == [1]
assert find_triple(60) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
assert find_triple(160) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


def find_double(n):
    temp_n = min(n, 40)
    doubles = [i for i in range(1, temp_n+1) if i * 2 <= temp_n]
    if n >= 50:
        doubles.append(25)
    return doubles


assert find_double(0) == []
assert find_double(1) == []
assert find_double(2) == [1]
assert find_double(3) == [1]
assert find_double(4) == [1,2]
assert find_double(40) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
assert find_double(49) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
assert find_double(50) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25]
assert find_double(170) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25]


def find_single(n):
    temp_n = min(n, 20)
    singles = [i for i in range(1, temp_n+1) if i <= temp_n]
    if n >= 25:
        singles.append(25)
    return singles

assert find_single(0) == []
assert find_single(1) == [1]
assert find_single(2) == [1,2]
assert find_single(3) == [1,2,3]
assert find_single(4) == [1,2,3,4]
assert find_single(20) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
assert find_single(24) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
assert find_single(25) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25]
assert find_single(170) == [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25]


assert sum(number_checkout(i) for i in range(1, 171)) == 42336
print(sum(number_checkout(i) for i in range(1, 100)))