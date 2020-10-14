import numpy as np


def calculate_cost(num, depth, cost, path):
    if num > LIMIT or cost[num] < depth:
        pass
    else:
        cost[num] = depth
        path[depth] = num
        for i in range(depth, -1, -1):
            calculate_cost(num+path[i], depth+1, cost, path)


def init_and_calculate():
    cost = np.zeros((LIMIT + 1), np.int64)
    path = np.zeros((LIMIT + 1), np.int64)
    for i in range(1, LIMIT + 1):
        cost[i] = 1000
    return cost, path


LIMIT = 200
cost, path = init_and_calculate()
calculate_cost(1, 0, cost, path)

print(sum(cost))
