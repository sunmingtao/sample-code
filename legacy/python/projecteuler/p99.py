import math
math.log10(2) * 11 > math.log10(3) * 7


text_file = open("p99.txt", "r")
lines = text_file.read().split('\n')
text_file.close()

max_num = 0
max_base = 0
max_exp = 0
math.log10(632382)*518061
for line in lines:
    line_split = line.split(',')
    base = int(line_split[0])
    exp = int(line_split[1])
    temp = math.log10(base) * exp
    if temp > max_num:
        max_num, max_base, max_exp = temp, base, exp

print (max_base, max_exp)
lines[0]