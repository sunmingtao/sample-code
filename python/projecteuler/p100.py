import math

math.sqrt(14281)

def cal(t):
    return 1 + 2 * t * (t-1)

def is_square(n):
    return int(math.sqrt(n)) ** 2 == n

start = 10 ** 12


for i in range(1, 50000000):
    s2 = math.sqrt(2)
    if is_square(cal(i)):
        print(i, s2 * i)

s2 = math.sqrt(2)
for i in range(start+100000000, start+100000000+1):
    s2_i = s2 * i
    #if 0.7071 < s2_i - int(s2-i) < 0.7072:
    print (i, s2 * i)