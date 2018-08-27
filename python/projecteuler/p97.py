limit = 7830457
result = 2
for i in range(1, limit):
    result *= 2
    if len(str(result)) > 10:
        result = int(str(result)[-10:])

print(result)