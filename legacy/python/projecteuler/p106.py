import itertools

candidate_set = (20,31,38,39,40,42,45)


def contains_small_set_large_sum(candidate_set):
    candidate_set = list(candidate_set)
    candidate_set.sort()
    for i in range(1, (len(candidate_set) - 1) // 2 + 1):
        small_sum = sum(candidate_set[0 : i+1])
        large_sum = sum(candidate_set[-i:])
        if small_sum <= large_sum:
            return True
    return False


assert not contains_small_set_large_sum((11, 17, 20, 22, 23, 24))
assert contains_small_set_large_sum((10, 17, 20, 22, 23, 24))
assert not contains_small_set_large_sum((20, 31, 38, 39, 40, 42, 45))
assert not contains_small_set_large_sum((157, 150, 164, 119, 79, 159, 161, 139, 158))


def has_same_total(set1, set2):
    return sum(set1) == sum(set2)


def contains_equal_sum(candidate_set):
    candidate_set = list(candidate_set)
    candidate_set.sort()
    for i in range(2, len(candidate_set) // 2 + 1):
        sub_sets = list(itertools.combinations(candidate_set, i))
        for _subset in sub_sets:
            remaining_set = set(candidate_set) - set(_subset)
            remaining_subsets = list(itertools.combinations(remaining_set, i))
            for _remaining_subset in remaining_subsets:
                if has_same_total(_subset, _remaining_subset):
                    return True
    return False

assert not contains_equal_sum((11, 17, 20, 22, 23, 24))
assert not contains_equal_sum((11, 18, 19, 20, 22, 25))
assert contains_equal_sum((42, 65, 75, 81, 84, 86, 87, 88))



assert not contains_equal_sum((157, 150, 164, 119, 79, 159, 161, 139, 158))
assert not contains_equal_sum((20, 31, 38, 39, 40, 42, 45))


text_file = open("projecteuler/p105.txt", "r")
lines = text_file.read().split('\n')
text_file.close()



total = 0
for line in lines:
    candidate_set = tuple(int(i) for i in line.split(','))
    if not contains_small_set_large_sum(candidate_set) and not contains_equal_sum(candidate_set):
        total += sum(candidate_set)
print(total)

list(itertools.combinations((3,5,6,7), 2))


sub_sets = list(itertools.combinations((1,2,3,4,5,6,7), 2))

def contains_duplicate_elements(set1, set2):
    return len(set(set1) | set(set2)) < len(set1) * 2


assert contains_duplicate_elements((1,2), (2,3))
assert not contains_duplicate_elements((1,2), (3,4))

def no_need_to_test(set1, set2):
    if max(set1) < min(set2):
        return True
    elif sum(n1 < n2 for n1, n2 in zip(set1, set2)) == len(set1):
        return True
    else:
        return sum(n1 > n2 for n1, n2 in zip(set1, set2)) == len(set1)


assert no_need_to_test((1,2), (3,4))
assert no_need_to_test((1,3), (2,4))
assert no_need_to_test((2,4), (1,3))
assert not no_need_to_test((1,4), (2,3))
assert no_need_to_test((1,2,4), (3,5,6))
assert no_need_to_test((3,5,6), (1,2,4))
assert not no_need_to_test((1,2,6), (3,4,5))


def num_need_test(sub_sets):
    total = 0
    for i in range(len(sub_sets)):
        for j in range(i):
            sub_set_1 = sub_sets[i]
            sub_set_2 = sub_sets[j]
            if (not contains_duplicate_elements(sub_set_1, sub_set_2)) and (not no_need_to_test(sub_set_1, sub_set_2)):
                total += 1
    return total

assert num_need_test(list(itertools.combinations((1,2,3,4), 2))) == 1
assert num_need_test(list(itertools.combinations((1,2,3,4,5,6,7), 2))) == 35
assert num_need_test(list(itertools.combinations((1,2,3,4,5,6,7), 3))) == 35

def num_need_test_for_n(n):
    total = 0
    seq = tuple(range(1,n+1))
    for i in range(2, n // 2 + 1):
        total += num_need_test(list(itertools.combinations(seq, i)))
    return total

assert num_need_test_for_n(4) == 1
assert num_need_test_for_n(7) == 70
print(num_need_test_for_n(12))

