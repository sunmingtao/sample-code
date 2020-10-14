
def is_pandigital(str):
    lst = list(str)
    lst.sort()
    return len(str) == 9 and ''.join(lst) == '123456789'

assert not is_pandigital('1437268955')
assert is_pandigital('143726895')

a = 1
b = 1
n = 3
while n < 10000000:
    f = a + b
    if is_pandigital(str(f % (10 ** 9))) and is_pandigital(str(f)[0:9]):
        print (n, f)
        break
    n += 1
    a = b
    b = f
    if n % 10000 == 0:
        print('Check n=', n)


