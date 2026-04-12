import math
import gmpy2
import time
import sympy

now = time.time()
result = []
for n in range (10, 150000001, 10):
    if n ** 2 % 3 == 1 and n ** 2 % 7 == 2 and n ** 2 % 13 != 0 and sympy.isprime(n ** 2 + 1) and sympy.isprime(n ** 2 + 3) and sympy.isprime(n ** 2 + 7) and sympy.isprime(n ** 2 + 9) and sympy.isprime(n ** 2 + 13) and sympy.isprime(n ** 2 + 27) \
            and not sympy.isprime(n ** 2 + 23) and not sympy.isprime(n ** 2 + 21) and not sympy.isprime(n ** 2 + 19) and not sympy.isprime(n ** 2 + 17):
        result.append(n)
        print(n)
print ('result=', sum(result))
print('time spent is {}'.format(time.time() - now))

