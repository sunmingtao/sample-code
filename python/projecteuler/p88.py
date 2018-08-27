import math

real_limit = 12000
limit =12200

arr = []
total_set = set()

for i in range(limit+1):
    arr.append(set())

def p(n):
    for a in range(2, int(math.sqrt(n))+1):
        if n % a == 0:
            b = n // a
            value = n - a - b + 1
            if value < real_limit:
                arr[n].add(value)
                total_set.add(value)
            for b_value in arr[b]:
                good_value = value + b_value
                if good_value < real_limit:
                    arr[n].add(good_value)
                    total_set.add(good_value)


for n in range(4, limit+1):
    p(n)



max(total_set)

good_numbers = []
good_set = set()
for i in range(len(arr)):
    for j in arr[i]:
        if j not in good_set:
            good_set.add(j)
            if i not in good_numbers:
                good_numbers.append(i)

print(good_numbers)

sum(good_numbers)