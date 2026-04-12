import numpy as np

NUM_UNIT = 50
MAX_BLOCKS = 12

matrix = np.zeros((MAX_BLOCKS+1, NUM_UNIT+1), dtype=np.int)

for b in range(0, MAX_BLOCKS + 1):
    for u in range(3, NUM_UNIT + 1):
        if b == 0:
            matrix[b, u] = 1
        elif b == 1:
            matrix[b, u] = (u - 1) * (u - 2) // 2
        else:
            matrix[b, u] = sum(matrix[b-1, i] * (u - 3 - i) for i in range(u - 4, 2, -1))


def ways(unit):
    return np.sum(matrix, axis=0)[unit]


assert ways(3) == 2
assert ways(4) == 4
assert ways(5) == 7
assert ways(6) == 11
assert ways(7) == 17
assert ways(8) == 27
assert ways(11) == 117
assert ways(49) == 10182505537
print (ways(50))