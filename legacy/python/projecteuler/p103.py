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
                    print (_subset, _remaining_subset)
                    return True
    return False

assert not contains_equal_sum((11, 17, 20, 22, 23, 24))
assert not contains_equal_sum((11, 18, 19, 20, 22, 25))
assert contains_equal_sum((42, 65, 75, 81, 84, 86, 87, 88))
assert not contains_equal_sum((157, 150, 164, 119, 79, 159, 161, 139, 158))
assert not contains_equal_sum((20, 31, 38, 39, 40, 42, 45))



