a, b, count = 3, 2, 0
for i in range(1000):
    if len(str(a)) > len(str(b)):
        count += 1
    temp_a = a + 2 * b
    b = a + b
    a = temp_a

count