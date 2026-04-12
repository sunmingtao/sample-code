def order(a, b):
    if a > b:
        return b, a
    return a, b

def results2_list(ass, bs):
    results = []
    for a in ass:
        for b in bs:
            results.extend(results2(a, b))
    return results

def results2(a, b):
    a, b = order(a, b)
    results = [a + b, b - a, a * b]
    if a != 0 and b % a == 0:
        results.append(b // a)
    return results

def results3(a, b, c):
    results = []
    results.extend(results2_list([a], results2(b, c)))
    results.extend(results2_list([b], results2(a, c)))
    results.extend(results2_list([c], results2(a, b)))
    return results

def results4(a, b, c, d):
    results = []
    results.extend(results2_list([a], results3(b, c, d)))
    results.extend(results2_list([b], results3(a, c, d)))
    results.extend(results2_list([c], results3(a, b, d)))
    results.extend(results2_list([d], results3(a, b, c)))

    results.extend(results2_list(results2(a,b), results2(c,d)))
    results.extend(results2_list(results2(a,c), results2(b,d)))
    results.extend(results2_list(results2(a,d), results2(b,c)))
    return results



def remove_duplicate(lst):
    new_list = list(set(lst))
    if 0 in new_list:
        new_list.remove(0)
    new_list.sort()
    return new_list

def longest(a,b,c,d):x
    lst = remove_duplicate(results4(a,b,c,d))
    i = 0
    while i < len(lst) and lst[i] == i + 1:
        i += 1
    return i

longest_num = []

for d in range(4, 10):
    for c in range(3, d):
        for b in range(2, c):
            for a in range(1, b):
                longest_ = longest(a,b,c,d)
                longest_num.append(longest_)
                if longest_ == 43:
                    print(longest_, a,b,c,d)

max(longest_num)

print(longest(1,2,3,4))
