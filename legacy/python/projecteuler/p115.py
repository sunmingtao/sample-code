import numpy as np

def init_matrix(minimal_unit, max_unit):
    max_blocks = max_unit // (minimal_unit + 1)
    matrix = np.zeros((max_blocks + 1, max_unit + 1), dtype=np.long)
    for b in range(0, max_blocks + 1):
        for u in range(minimal_unit, max_unit + 1):
            if b == 0:
                matrix[b, u] = 1
            elif b == 1:
                matrix[b, u] = (u - (minimal_unit - 2)) * (u - (minimal_unit - 1)) // 2
            else:
                matrix[b, u] = sum(matrix[b-1, i] * (u - minimal_unit - i) for i in range(u - minimal_unit - 1, minimal_unit - 1, -1))
    return matrix



def ways(minimal_unit, unit):
    matrix = init_matrix(minimal_unit, max_unit=unit+1)
    return np.sum(matrix, axis=0)[unit]


assert ways(3, 3) == 2
assert ways(3, 4) == 4
assert ways(3, 5) == 7
assert ways(3, 6) == 11
assert ways(3, 7) == 17
assert ways(3, 8) == 27
assert ways(3, 11) == 117
assert ways(3, 29) == 673135
assert ways(3, 30) == 1089155
assert ways(10, 56) == 880711
assert ways(10, 57) == 1148904
assert ways(3, 49) == 10182505537

for i in range(50, 200):
    if ways(50, i) > 1000000:
        print (i)
        break