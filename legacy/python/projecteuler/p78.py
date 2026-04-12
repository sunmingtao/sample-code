import math

ONE_MILL = 1000000


def fm(k):
    return k * (3 * k - 1) // 2


assert fm(1) == 1
assert fm(-1) == 2
assert fm(2) == 5
assert fm(-2) == 7
assert fm(3) == 12
assert fm(-3) == 15


def get_fn_item(n):
    return 0 if n < 0 else fn_list[n]


def fn(n):
    total = 0
    k = 1
    while n - fm(k) >= 0:
        total += int((-1) ** (k + 1)) * get_fn_item(n - fm(k)) + int((-1) ** (-k + 1)) * get_fn_item(n - fm(-k))
        k += 1
    return total


fn_list = [1]
assert fn(1) == 1
fn_list = [1, 1]
assert fn(len(fn_list)) == 2
fn_list = [1, 1, 2, 3, 5, 7, 11, 15, 22]
assert fn(len(fn_list)) == 30
fn_list = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627, 792, 1002]
assert fn(len(fn_list)) == 1255
fn_list = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231, 297, 385, 490, 627, 792, 1002, 1255, 1575, 1958, 2436, 3010, 3718, 4565]
assert fn(len(fn_list)) == 5604


fn_list = [1]


def calculate_fn():
    while True:
        n = len(fn_list)
        _fn = fn(n) % ONE_MILL
        if _fn == 0:
            print ('Found', n)
            break
        else:
            fn_list.append(_fn)


calculate_fn()


