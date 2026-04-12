import math
import gmpy2
import time
import sympy
import numpy as np

now = time.time()

def cut(comb, a):
    temp = list(comb).copy()
    temp[a]-=1
    for i in range(a+1, len(temp)):
        temp[i]+=1
    return tuple(temp)


assert cut((1,1,1,1), 0) == (0,2,2,2)
assert cut((1,1,1,1), 1) == (1,0,2,2)
assert cut((1,1,1,1), 2) == (1,1,0,2)
assert cut((1,1,1,1), 3) == (1,1,1,0)

def find_value(comb):
    return sum(comb[i] * 2 ** (len(comb) - i - 1) for i in range(len(comb)))

assert find_value((1,1,1,1)) == 15
assert find_value((0,2,2,2)) == 14
assert find_value((1,0,2,2)) == 14
assert find_value((1,1,0,2)) == 14
assert find_value((1,1,1,0)) == 14



def get_next_round_result(prev_round_result):
    result = {}
    for comb, p in prev_round_result.items():
        for i in range(len(comb)):
            if comb[i] != 0:
                after_cut_comb = cut(comb, i)
                prob = comb[i] / sum(comb)
                if not after_cut_comb in result:
                    result[after_cut_comb] = prob * p
                else:
                    result[after_cut_comb] += prob * p
    assert math.fabs(sum(p for _, p in result.items()) - 1.0) < 0.0000001
    return result


result = get_next_round_result(get_next_round_result({(1,1,1,1) : 1}))
assert result[(0,2,2,1)] == 1/6
assert result[(1,0,2,1)] == 1/12 + 1/10
assert result[(1,1,0,1)] == 1/12 + 1/8
assert result[(0,1,3,3)] == 1/12 + 1/20
assert result[(0,2,1,3)] == 1/12 + 1/16
assert result[(1,0,1,3)] == 1/10 + 1/16

total = 0
result = {(1,1,1,1) : 1}
for i in range(13):
    result = get_next_round_result(result)
    for k, v in result.items():
        if sum(k) == 1:
            total += v

print (total)


print('time spent is {}'.format(time.time() - now))

