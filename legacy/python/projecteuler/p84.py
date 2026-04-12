import random

def roll_dice():
    limit = 4
    a = random.randint(1, limit)
    b = random.randint(1, limit)
    return a,b



cc_cards = [-1 for i in range(16)]
cc_cards[0]=0
cc_cards[1]=10

random.shuffle(cc_cards)


ch_cards = [-1 for i in range(16)]
ch_cards[0] = 0
ch_cards[1] = 10
ch_cards[2] = 11
ch_cards[3] = 24
ch_cards[4] = 39
ch_cards[5] = 5
ch_cards[6] = 45
ch_cards[7] = 45
ch_cards[8] = 50
ch_cards[9] = -3

random.shuffle(ch_cards)


board= [0 for i in range(40)]
index = 0
double_count = 0
moves = 0
cc_card_index = 0
ch_card_index = 0

def move():
    global moves
    global double_count
    global index
    moves += 1
    a, b = roll_dice()
    if a == b:
        double_count += 1
        if double_count == 3:
            double_count = 0
            index = 10
            board[index] += 1
        else:
            normal_move(a + b)
    else:
        double_count = 0
        normal_move(a+b)

def normal_move(steps):
    global index
    global cc_card_index
    global ch_card_index
    index += steps
    index %= 40
    if index == 30:
        index = 10
    elif index == 2 or index == 17 or index == 33:
        cc_value = cc_cards[cc_card_index]
        cc_card_index += 1
        cc_card_index %= 16
        if cc_value == -1:
            pass
        else:
            index = cc_value
    elif index == 7 or index == 22 or index == 36:
        ch_value = ch_cards[ch_card_index]
        ch_card_index += 1
        ch_card_index %= 16
        if ch_value == -1:
            pass
        elif ch_value == -3:
            normal_move(-3)
            return
        elif ch_value == 45: # Go to next railway company
            index = get_next_railway(index)
        elif ch_value == 50: # Go to next utility
            index = get_next_utility(index)
        else:
            index = ch_value
    board[index] += 1

def get_next_railway(local_index):
    if local_index == 7:
        return 15
    elif local_index == 22:
        return 25
    else:
        return 5


def get_next_utility(local_index):
    if local_index == 7 or local_index == 36:
        return 12
    else:
        return 28


for i in range(500000):
    move()

print(board[0]/moves)
print(board[10]/moves)
print(board[24]/moves)
print(board[2]/moves)
print(board[17]/moves)
print(board[33]/moves)

max(board)

board_copy = board.copy()
board_copy.sort()

board.index(16414)


101524













