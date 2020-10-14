import numpy as np

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


def find_num_answers(n):
    count = 0
    for i in range(n+1, n*2+1):
        if (n * i / (i - n)).is_integer():
            count += 1
    return count


assert find_num_answers(4) == 3
assert find_num_answers(1260) == 113
assert find_num_answers(1260*11) == 338
assert find_num_answers(1260*11*2) == 473
assert find_num_answers(1260*11*13) == 1013
assert find_num_answers(6126120) == 4253
assert find_num_answers(12252240) == 5468
assert find_num_answers(30630600) == 7088
assert find_num_answers(61261200) == 9113

assert find_num_answers(2677114440) == 12758



seq = [2, 2, 1, 1]
primes = get_all_primes(100)

def get_n(seq):
    local_seq = np.array(seq)
    local_primes = np.array(primes[:len(seq)])
    return np.product(local_primes ** local_seq)


assert get_n([2, 2, 1, 1]) == 1260


def find_num_answer_from_seq(seq):
    local_seq = np.array(seq) * 2 + 1
    return (np.product(local_seq) + 1) // 2


def find_number_and_num_answer_from_seq(seq):
    return get_n(seq), find_num_answer_from_seq(seq)


assert find_num_answer_from_seq([2, 2, 1, 1]) == 113
assert find_num_answer_from_seq([2, 2, 1, 1, 1, 1]) == 1013
assert find_num_answer_from_seq([2, 2, 1, 1, 1, 1, 1]) == 3038
assert find_num_answer_from_seq([2, 2, 2, 2, 2, 2, 2, 2, 2, 1]) == 2929688
assert find_num_answer_from_seq([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 2214338
assert find_num_answer_from_seq([2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 3690563
assert find_num_answer_from_seq([3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 3100073
assert find_num_answer_from_seq([4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 3985808
assert find_num_answer_from_seq([5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 4871543
assert find_num_answer_from_seq([3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 4340102
assert find_num_answer_from_seq([2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == 6643013

assert find_number_and_num_answer_from_seq([2, 2, 1, 1]) == (1260, 113)
assert find_number_and_num_answer_from_seq([3, 2, 1, 1]) == (2520, 158)
assert find_number_and_num_answer_from_seq([3, 3, 1, 1]) == (7560, 221)
assert find_number_and_num_answer_from_seq([3, 2, 2, 1]) == (12600, 263)
assert find_number_and_num_answer_from_seq([2, 2, 1, 1, 1]) == (13860, 338)
assert find_number_and_num_answer_from_seq([2, 2, 1, 1, 1, 1]) == (180180, 1013)
assert find_number_and_num_answer_from_seq([3, 2, 1, 1, 1, 1]) == (360360, 1418)
assert find_number_and_num_answer_from_seq([3, 2, 1, 1, 1, 1, 1]) == (6126120, 4253)
assert find_number_and_num_answer_from_seq([4, 2, 1, 1, 1, 1, 1]) == (12252240, 5468)
assert find_number_and_num_answer_from_seq([3, 2, 2, 1, 1, 1, 1]) == (30630600, 7088)
assert find_number_and_num_answer_from_seq([4, 2, 2, 1, 1, 1, 1]) == (61261200, 9113)
assert find_number_and_num_answer_from_seq([3, 2, 1, 1, 1, 1, 1, 1]) == (116396280, 12758)
assert find_number_and_num_answer_from_seq([3, 2, 1, 1, 1, 1, 1, 1, 1]) == (2677114440, 38273)
assert find_number_and_num_answer_from_seq([3, 2, 2, 1, 1, 1, 1, 1, 1]) == (13385572200, 63788)
assert find_number_and_num_answer_from_seq([3, 3, 2, 1, 1, 1, 1, 1, 1]) == (40156716600, 89303)
assert find_number_and_num_answer_from_seq([3, 3, 2, 1, 1, 1, 1, 1, 1, 1]) == (1164544781400, 267908)
assert find_number_and_num_answer_from_seq([3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]) == (36100888223400, 803723)
assert find_number_and_num_answer_from_seq([4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]) == (72201776446800, 1033358)
assert find_number_and_num_answer_from_seq([4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (2671465728531600, 3100073)
assert find_number_and_num_answer_from_seq([5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (5342931457063200, 3788978)
assert find_number_and_num_answer_from_seq([6, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (10685862914126400, 4477883)
assert find_number_and_num_answer_from_seq([4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (8014397185594800, 3985808)
assert find_number_and_num_answer_from_seq([3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]) == (9350130049860600, 4018613) #Haha

assert find_number_and_num_answer_from_seq([5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (14604012649306080, 4871543)
assert find_number_and_num_answer_from_seq([3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (10953009486979560, 4340102)
assert find_number_and_num_answer_from_seq([4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == (7302006324653040, 3985808)













