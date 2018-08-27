def same_digit(num1, num2):
    num1_str = str(num1)
    num2_str = str(num2)
    for i in num1_str:
        if len(num1_str) != len(num2_str) or num1_str.count(i) != num2_str.count(i):
            return False
    return True

upper_bound = 1000000

candidates_1 = [i for i in range(1, upper_bound) if int(str(i)[0:2]) < 17]


def get_candidate(candidates, multiple):
    candidate_ = []
    for i in candidates:
        if same_digit(i, i * multiple):
            candidate_.append(i)
    return candidate_

candidates_2 = get_candidate(candidates_1, 2)

candidates_3 = get_candidate(candidates_2, 3)

candidates_4 = get_candidate(candidates_3, 4)

candidates_5 = get_candidate(candidates_4, 5)

candidates_6 = get_candidate(candidates_5, 6)

