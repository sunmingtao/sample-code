def find_num_answers(n):
    count = 0
    for i in range(n+1, n*2+1):
        if (n * i / (i - n)).is_integer():
            #print (i, n * i // (i - n))
            count += 1
    return count

assert find_num_answers(4) == 3


largest = 0
for i in range(83610, 1000000, 10):
    num_answers = find_num_answers(i)
    if num_answers > largest:
        largest = num_answers
        print(i, num_answers)
        if num_answers > 1000:
            break
