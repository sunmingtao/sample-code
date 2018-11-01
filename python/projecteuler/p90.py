import itertools


digits = [0,1,2,3,4,5,6,7,8,9]

dice_list = list(itertools.combinations(digits, 6))

square_numbers = ['01', '04', '09', '16', '25', '36', '49', '64', '81']

def can_form_all_square_numbers(dice1, dice2):
    for number in square_numbers:
        if not can_form_number(dice1, dice2, number):
            return False
    return True


def can_form_number(dice1, dice2, number):
    for d1 in dice1:
        for d2 in dice2:
            if can_digit_form_number(d1, d2, number):
                return True
    return False


def can_digit_form_number(d1, d2, number):
    if str(d1) + str(d2) == str(number) or str(d2) + str(d1) == str(number):
        return True
    elif d1 == 6 or d1 == 9:
        return str(exchange_69(d1)) + str(d2) == str(number) or str(d2) + str(exchange_69(d1)) == str(number)
    elif d2 == 6 or d2 == 9:
        return str(d1) + str(exchange_69(d2)) == str(number) or str(exchange_69(d2)) + str(d1) == str(number)
    else:
        return False


def exchange_69(digit):
    if digit == 6:
        return 9
    elif digit == 9:
        return 6
    else:
        return digit

assert can_digit_form_number(0, 4, '04')
assert can_digit_form_number(4, 0, '04')
assert not can_digit_form_number(4, 1, '04')
assert can_digit_form_number(6, 0, '09')
assert can_digit_form_number(9, 0, '09')
assert not can_digit_form_number(9, 6, '09')

assert can_form_number((0, 5, 6, 7, 8, 9), (1, 2, 3, 4, 8, 9), '09')
assert can_form_number((0, 5, 6, 7, 8, 9), (1, 2, 3, 4, 8, 6), '09')
assert not can_form_number((0, 5, 6, 7, 8, 9), (1, 2, 3, 4, 8, 7), '09')

assert can_form_all_square_numbers((0, 5, 6, 7, 8, 9), (1, 2, 3, 4, 8, 9))
assert can_form_all_square_numbers((0, 5, 6, 7, 8, 9), (1, 2, 3, 4, 8, 6))
assert not can_form_all_square_numbers((0, 5, 6, 7, 8, 9), (1, 2, 3, 4, 8, 7))

counter = 0
for dice1 in dice_list:
    for dice2 in dice_list:
        if can_form_all_square_numbers(dice1, dice2):
            print(dice1, dice2)
            counter += 1

print (counter // 2)