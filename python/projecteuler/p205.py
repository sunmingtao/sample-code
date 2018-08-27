bucket4 = [0 for _ in range(37)]
for a in range(1, 5):
    for b in range(1,5):
        for c in range(1,5):
            for d in range(1,5):
                for e in range(1, 5):
                    for f in range(1, 5):
                        for g in range(1, 5):
                            for h in range(1, 5):
                                for i in range(1, 5):
                                    bucket4[a+b+c+d+e+f+g+h+i] += 1

bucket6 = [0 for _ in range(37)]
for a in range(1,7):
    for b in range(1, 7):
        for c in range(1, 7):
            for d in range(1, 7):
                for e in range(1, 7):
                    for f in range(1, 7):
                        bucket6[a + b + c + d + e + f] += 1

print(bucket6)

total_win = 0
for i4 in range(9, 37):
    v4 = bucket4[i4]
    for i6 in range(6, 37):
        v6 = bucket6[i6]
        if i4 > i6:
            total_win += v4 * v6

print(total_win)

total_win/(4 ** 9 * 6 ** 6)