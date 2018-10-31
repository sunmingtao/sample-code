text_file = open("projecteuler/p98.txt", "r")
lines = text_file.read().replace('"', '').split(',')
text_file.close()

square_nums = [i ** 2 for i in range(1, 31623)]

def char_num(word):
    return {ch: word.count(ch) for ch in word}

def is_anagram_pair(word1, word2):
    return len(word1) == len(word2) and char_num(word1.upper()) == char_num(word2.upper())

assert is_anagram_pair('RACE', 'CARE')
assert not is_anagram_pair('RACE', 'ACE')

def get_anagram_pairs():
    pairs = []
    for i in range(len(lines)):
        for j in range(i):
            if is_anagram_pair(lines[i], lines[j]):
                pairs.append((lines[i], lines[j]))
    return pairs


anagram_pairs = get_anagram_pairs()


def get_len_n_square_nums(n):
    return [str(num) for num in square_nums if len(str(num)) == n]


def get_candidate_num(word, ch_d_dict):
    a = []
    for w in word:
        a.append(ch_d_dict[w])
    return ''.join(a)


def char_digit_dict(word, num):
    if len(word) != len(num):
        return None
    else:
        keys = []
        values = []
        dictionary = {}
        for char, digit in zip(word, num):
            if char in keys:
                if dictionary[char] != digit:
                    return None
            elif digit in values:
                return None
            else:
                keys.append(char)
                values.append(digit)
                dictionary[char] = digit
        return dictionary


assert char_digit_dict('EDGE', '98791') == None
assert char_digit_dict('EDGE', '9981') == None
assert char_digit_dict('EDGE', '9876') == None
assert char_digit_dict('EDGE', '9879') == {'E': '9', 'D': '8', 'G': '7'}

for word1, word2 in anagram_pairs:
    len_n_square_nums = get_len_n_square_nums(len(word1))
    for num in len_n_square_nums:
        _dict = char_digit_dict(word1, num)
        if _dict:
            candidate_num = get_candidate_num(word2, _dict)
            if candidate_num in len_n_square_nums:
                print('Found pair {}, {}, {}, {}'.format(word1, word2, num, candidate_num))



