import math

def find_abcd(m, n):
    a = m ** 2 - n ** 2
    b = 2 * m * n
    c = m ** 2 + n ** 2
    d = a+b+c
    return a,b,c,d

all_set = set()
duplicate_set = set()

upper_limit = 1500000

for m in range(2, 1000):
    for n in range(1, m):
        if ((m % 2 == 0 and n % 2 != 0) or (m % 2 != 0 and n % 2 == 0)) and math.gcd(m, n) == 1:
            a,b,c,d = find_abcd(m,n)
            if a+b+c <= upper_limit:
                temp_d = d
                while temp_d <= upper_limit:
                    if temp_d in all_set:
                        duplicate_set.add(temp_d)
                    else:
                        all_set.add(temp_d)
                    temp_d += d
            else:
                break


print(len(all_set))
print(len(duplicate_set))

120 in duplicate_set

find_abcd(864,1)