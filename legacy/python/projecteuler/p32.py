def has_duplicate_digit(num):
    return len(set(str(num))) != len(str(num))


no_duplicate = [i for i in range(1, 9999) if not has_duplicate_digit(i)]

len(no_duplicate)

all_num = set()
for i in no_duplicate:
    for j in no_duplicate:
        concat = str(i)+str(j)+str(i*j)
        if len(set(concat)) == 9 and len(concat) == 9 and concat.find('0') == -1:
            print(i, j, i*j)
            all_num.add(i*j)
        if len(str(i)) + len(str(j)) + len(str(i*j)) > 9:
            break

print(all_num)
print(sum(all_num))

'123456789'.strip('5672341')