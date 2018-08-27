start = 1010101020
end =   1389026620

def extract_odd_digit(n):
    result = ''
    str_n = str(n)
    for i in range(0, len(str_n), 2):
        result += str_n[i]
    return result


for i in range(start, end+1, 10):
    num = i ** 2
    if extract_odd_digit(num) == '1234567890':
        print(i)
        break


