17 * 1 = 017
17 * 2 = 034
17 * 3 = 051
17 * 4 - 068
...
17 * 17 = 289
...
17 * 58 = 986

for i in range (1, 59):
    print (i * 17)




def no_duplicate(num):
    num_str = str(num).zfill(3)
    return len(set(num_str)) == 3

def no_duplicate_for_str(num):
    return len(set(num)) == len(num)

no_duplicate(19)

def create_list(num):
    return [str(i * num).zfill(3) for i in range(1, 1000//num+1) if no_duplicate(i * num)]

list_17 = create_list(17)
list_13 = create_list(13)
list_11 = create_list(11)
list_7 = create_list(7)
list_5 = create_list(5)
list_3 = create_list(3)
list_2 = create_list(2)

def all_concat(list1, list2):
    result = []
    for l1 in list1:
        for l2 in list2:
            if l1[-2:] == l2[0:2] and no_duplicate_for_str(l1[0:1]+l2):
                result.append(l1[0:1]+l2)
    return result

result = all_concat(list_13, list_17)

result = all_concat(list_11, result)

result = all_concat(list_7, result)

result = all_concat(list_5, result)

result = all_concat(list_3, result)

result = all_concat(list_2, result)

len(result)

sum(int(i) for i in ['4106357289', '4130952867', '4160357289', '1406357289', '1430952867', '1460357289'])
