for a in range(1, 10):
    for b in range(1, 10):
        for c in range(1, 10):
            p = int(str(a)+str(b)) / int(str(b)+str(c))
            q = a / c
            if p == q and a != b:
                print(a,b,c)
