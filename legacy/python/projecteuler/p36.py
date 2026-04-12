def is_palindromic(s):
    return s[::-1] == s


sum(i for i in range(1, 1000000) if is_palindromic(bin(i)[2:]) and is_palindromic(str(i)))
