text_file = open("p89.txt", "r")
lines = text_file.read().split()
text_file.close()


roman = {'I' : 1, 'V' : 5, 'X' : 10, 'L' : 50, 'C' : 100, 'D' : 500, 'M' : 1000, 'IV':4, 'IX':9, 'XL':40, 'XC':90, 'CD':400, 'CM':900}

def to_num(roman_chars):
    sum = 0
    skip_next = False
    char_len = len(roman_chars)
    for i in range(char_len):
        if skip_next:
            skip_next = False
            continue
        char = roman_chars[i]
        char_group = char
        if i < char_len - 1:
            next_char = roman_chars[i+1]
            if roman[next_char] > roman[char]:
                char_group = char + next_char
                skip_next = True
        sum += roman[char_group]
    return sum


to_char(to_num('MCDLXXXIX'))

original_total_min_char_len = 0
total_min_char_len = 0
for i in lines:
    num_ = to_num(i)
    min_char = to_char(num_)
    total_min_char_len += len(min_char)
    original_total_min_char_len += len(i)
    num_2 = to_num(min_char)
    if num_ != num_2:
        print ('bad', i, num_, min_char, to_num(min_char))
    else:
        print ('good', i, num_, min_char, to_num(min_char))

print(original_total_min_char_len)
print(total_min_char_len)


def to_char(num):
    num_str = str(num)
    num_len = len(num_str)
    ch = ''
    for i in range(num_len, 0, -1):
        digit = int(num_str[num_len-i])
        ch += digit_to_char(digit, i)
    return ch

to_char(2399)

def digit_to_char(digit, index):
    if index == 4:
        return 'M' * digit
    elif index == 3:
        if digit == 9:
            return 'CM'
        elif digit == 4:
            return 'CD'
        elif digit >= 5:
            return 'D' + 'C' * (digit - 5)
        else:
            return 'C' * digit
    elif index == 2:
        if digit == 9:
            return 'XC'
        elif digit == 4:
            return 'XL'
        elif digit >= 5:
            return 'L' + 'X' * (digit - 5)
        else:
            return 'X' * digit
    else:
        if digit == 9:
            return 'IX'
        elif digit == 4:
            return 'IV'
        elif digit >= 5:
            return 'V' + 'I' * (digit - 5)
        else:
            return 'I' * digit
