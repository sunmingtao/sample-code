'''
If p is the perimeter of a right angle triangle with integral length sides, {a,b,c}, there are exactly three solutions for p = 120.

{20,48,52}, {24,45,51}, {30,40,50}

For which value of p ≤ 1000, is the number of solutions maximised?
'''

20 ** 2 + 48 ** 2 == 52 ** 2



def find_p (p):
    count = 0
    for a in range(1, int(p / 2)):
        for b in range(1, int(p / 2)):
            c = p - a - b
            lst = [a,b,c]
            lst.sort()
            if lst[0] ** 2 + lst[1] ** 2 == lst[2] ** 2:
                count+=1
    return int(count/6)

max_count = 0

max_p = 1000
good_p = 0
for i in range(1, max_p+1):
    temp_count = find_p(i)
    if max_count < temp_count:
        max_count, good_p = temp_count, i

print(max_count, good_p)
find_p(120)

