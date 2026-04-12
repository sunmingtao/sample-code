import math

def get_all_primes(n):
    arr = [True for n in range(n+1)]
    arr[0], arr[1] = False, False
    arr[2] = True
    index = 2
    while index < len(arr):
        for j in range(index * 2, n+1, index):
            arr[j] = False
        index += 1
        while index < len(arr) and not arr[index]:
            index += 1
    return [i for i in range(n+1) if arr[i]]

all_primes = get_all_primes(1000000)
len(all_primes)

def r(n):
    prime = all_primes[n-1]
    if n % 2 == 0:
        return 2
    else:
        return 2 * n * prime % (prime ** 2)

assert r(3) == 5
assert r(7036) < 10 ** 9
assert r(7037) > 10 ** 9

for i in range(7037, len(all_primes)):
    if r(i) > 10 ** 10:
        print (i)
        break

