num_set = set()
for i in range(10, 295246):
    if sum(int(j) ** 5 for j in str(i)) == i:
        num_set.add(i)

print(num_set)
print(sum(num_set))

