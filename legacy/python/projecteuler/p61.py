def p3(n):
    return n * (n+1) // 2

def p4(n):
    return n ** 2

def p5(n):
    return n * (3 * n - 1) // 2

def p6(n):
    return n * (2 * n - 1)

def p7(n):
    return n * (5 * n - 3) // 2

def p8(n):
    return n * (3 * n - 2)

def is_same_two_digit(num1, num2):
    num1_str = str(num1)
    num2_str = str(num2)
    return num1_str[-2:] == num2_str[0:2]


def init_polygonal_numbers():
    polygonal_numbers = []
    polygonal_numbers.append([p3(n) for n in range(150) if p3(n) >= 1000 and p3(n) <= 9999])
    polygonal_numbers.append([p4(n) for n in range(150) if p4(n) >= 1000 and p4(n) <= 9999])
    polygonal_numbers.append([p5(n) for n in range(150) if p5(n) >= 1000 and p5(n) <= 9999])
    polygonal_numbers.append([p6(n) for n in range(150) if p6(n) >= 1000 and p6(n) <= 9999])
    polygonal_numbers.append([p7(n) for n in range(150) if p7(n) >= 1000 and p7(n) <= 9999])
    polygonal_numbers.append([p8(n) for n in range(150) if p8(n) >= 1000 and p8(n) <= 9999])
    return polygonal_numbers

def search(result, remaining_index):
    if len(result) == result_len:
        if is_same_two_digit(result[-1][1], result[0][1]):
            print(sum(n[1] for n in result))
    else:
        for i in remaining_index:
            for j in polygonal_numbers[i]:
                if is_same_two_digit(result[-1][1], j):
                    new_result = result.copy()
                    new_result.append((i, j))
                    new_remaining_index = remaining_index.copy()
                    new_remaining_index.remove(i)
                    search(new_result, new_remaining_index)


result_len = 6
polygonal_numbers = init_polygonal_numbers()

for p in polygonal_numbers[5]:
    search([(5, p)], [0,1,2,3,4])


result_len = 3
polygonal_numbers = init_polygonal_numbers()

for p in polygonal_numbers[2]:
    search([(2, p)], [0,1])

