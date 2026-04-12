def is_perm(num1, num2):
    num1_str = str(num1)
    num2_str = str(num2)
    for c in '0123456789':
        if (num1_str.count(c) != num2_str.count(c)):
            return False
    return True

start = 465
end = 1001

#start = 216
#end = 465

#start = 1000
#end = 2155

#start = 2155
#end = 4642

start = 4642
end = 10000
for i in range(start,end):
    perm_list = [i]
    for j in range(start, end):
        if j > i:
            if is_perm(i**3, j**3):
                perm_list.append(j)
    if len(perm_list) >= 3:
        print(perm_list)


for i in range(4642, 10000):
    print(i, i**3)

5027**3