import math
upper_bound = 500000
max_size = 0
max_m, max_n = 0, 0
for n in range(5, upper_bound):
    m_limit = min(int(233329/99998*n), 10000000)
    for m in range (int(7*n/3+1), m_limit):
        if n/m > max_size:
            max_size, max_m, max_n = n/m, m, n

print(max_size, max_m, max_n)

for n in range(5, upper_bound):
    print(n)

