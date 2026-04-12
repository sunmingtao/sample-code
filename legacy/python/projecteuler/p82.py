import numpy as np

size = 80

text_file = open("p82.txt", "r")
lines = text_file.read().replace('\n', ',').split(',')
text_file.close()

arr = np.array(lines).reshape(size,size).astype(int).tolist()

def update_min_value(i, j):
    if j == 0:
        min_values[i][j] = arr[i][j]
    else:
        min_ = 999999999
        for k in range(size):
            temp_value = get_k_sum(i, j, k) + min_values[k][j-1] + arr[i][j]
            if temp_value < min_:
                min_ = temp_value
        min_values[i][j] = min_

def get_k_sum(i, j, k):
    total = 0
    if k <= i:
        step = 1
    else:
        step = -1
    for p in range(k, i, step):
        total += arr[p][j]
    return total

min_values = np.zeros((size, size)).astype(int).tolist()


for j in range(size):
    for i in range(size):
        update_min_value(i, j)

min_z = 9999999
for z in min_values:
    temp_min = z[-1]
    if temp_min < min_z:
        min_z = temp_min

print(min_z)

min(min_values[-1])


