import numpy as np

size = 80

text_file = open("p81_2.txt", "r")
lines = text_file.read().replace('\n', ',').split(',')
text_file.close()

arr = np.array(lines).reshape(size,size).tolist()


min_values = np.zeros((size, size)).tolist()
type(min_values[0][0])

for i in range(0, size):
    for j in range(0, size):
        if i == 0 and j == 0:
            min_values[i][j] = int(arr[i][j])
        elif i == 0:
            min_values[i][j] = int(arr[i][j]) + int(min_values[i][j-1])
        elif j == 0:
            min_values[i][j] = int(arr[i][j]) + int(min_values[i-1][j])
        else:
            min_values[i][j] = int(min(min_values[i][j-1], min_values[i-1][j])) +  int(arr[i][j])

for i in range(0, size):
    for j in range(0, size):
        print(min_values[i][j])

min_values[size-1][size-1]

big_numer = 10000000
arr[79][79]

def min_path(i, j):
    if i == 0 and j == 0:
        return int(arr[i][j])
    if i == -1 or j == -1:
        return big_numer

    return int(min(min_path(i-1, j), min_path(i, j-1)))+int(arr[i][j])


min_path(20,20)
