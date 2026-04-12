import math

upper_bound = 1000001

def is_relative_prime(num1, num2):
    return math.gcd(num1, num2) == 1

def phi(n):
    return len([i for i in range(1,n) if i % 2 != 0 and i % 3 != 0 and i % 5 != 0 and is_relative_prime(i, n)])


max(i/phi(i) for i in range(2,upper_bound) if i % 2 == 0)


max_phi = 0
max_i = 0
for i in range(2310,upper_bound,2310):

    phi_i = phi(i)
    if i/phi_i > max_phi:
        max_phi = i/phi_i
        max_i = i

print(max_i, max_phi)

print(phi(4620))

for i in range(2310,upper_bound,2310):

    print (i)