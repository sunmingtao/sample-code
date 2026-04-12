import math

def r(a, n):
   return ((a - 1) ** n + (a + 1) ** n) % (a ** 2)

assert r(7, 3) == 42
r(997, 497)
r(997, 1495)

def max_r(a):
    return max(r(a, n) for n in range(2, 1001))

max_r(3)
assert max_r(7) == 42
assert max_r(1000) == 998000

print (sum(max_r(a) for a in range(3, 1001)))

[max_r(a) for a in range(3, 1001)]


def max_r2(a):
    if a % 2 == 0:
        return a ** 2 - 2 * a
    else:
        return a ** 2 - a


assert max_r2(7) == 42
assert max_r2(1000) == 998000

print (sum(max_r2(a) for a in range(3, 1001)))

for a in range(3, 1000):
    if not max_r(a) == max_r2(a):
        print (a)


max_r(997)
max_r2(997)

for n in range(2, 2001):
    print (n, r(997, n))

