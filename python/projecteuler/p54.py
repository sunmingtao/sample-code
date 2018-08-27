text_file = open("p54.txt", "r")
lines = text_file.read().split('\n')
text_file.close()


def get_rank_value(rank):
    if rank.isnumeric():
        return int(rank)
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    elif rank == 'A':
        return 14
    raise ValueError('Unknow rank: '+rank)


def get_ranks(hand):
    rank_list = [get_rank_value(i[0]) for i in hand]
    rank_list.sort(reverse=True)
    return rank_list

def get_suits(hand):
    return [i[1] for i in hand]

def get_straight(ranks):
    if len(set(ranks)) == 5 and ranks[0] - ranks[4] == 4:
        return ranks[0]


def get_flush(ranks, suits):
    if len(set(suits)) == 1:
        return ranks[0]


def get_four_of_a_kind(ranks):
    rank_set = set(ranks)
    if len(rank_set) == 2:
        for i in rank_set:
            if ranks.count(i) == 4:
                return i

def get_full_house(ranks):
    rank_set = set(ranks)
    if len(rank_set) == 2:
        for i in rank_set:
            if ranks.count(i) == 3:
                return i

def get_three_of_a_kind(ranks):
    rank_set = set(ranks)
    if len(rank_set) == 3:
        for i in rank_set:
            if ranks.count(i) == 3:
                return i

def get_two_pair(ranks):
    rank_set = set(ranks)
    if len(rank_set) == 3:
        pair = []
        for i in rank_set:
            if ranks.count(i) == 2:
                pair.append(i)
            else:
                kicker = i
        pair.sort(reverse=True)
        return pair, kicker

def get_one_pair(ranks):
    rank_set = set(ranks)
    if len(rank_set) == 4:
        kicker = []
        for i in rank_set:
            if ranks.count(i) == 2:
                pair = i
            else:
                kicker.append(i)
        kicker.sort(reverse=True)
        return pair, kicker

def get_high_card(ranks):
    rank_set = set(ranks)
    if len(rank_set) == 5:
        return ranks

def get_hand_power(hand):
    ranks = get_ranks(hand)
    suits = get_suits(hand)
    flush = get_flush(ranks, suits)
    straight = get_straight(ranks)
    if flush is not None and straight is not None:
        return 9, straight
    four_of_a_kind = get_four_of_a_kind(ranks)
    if four_of_a_kind is not None:
        return 8, four_of_a_kind
    full_house = get_full_house(ranks)
    if full_house is not None:
        return 7, full_house
    elif flush is not None:
        return 6, flush
    elif straight is not None:
        return 5, straight
    three_of_a_kind = get_three_of_a_kind(ranks)
    if three_of_a_kind is not None:
        return 4, three_of_a_kind
    two_pair = get_two_pair(ranks)
    if two_pair is not None:
        return 3, two_pair
    one_pair = get_one_pair(ranks)
    if one_pair is not None:
        return 2, one_pair
    high_card = get_high_card(ranks)
    if high_card is not None:
        return 1, high_card
    raise ValueError('Unknown hand power: '+str(hand))

test_hand = 'TC JC JC 8C 8H'.split()
get_hand_power(test_hand)

player1_win = 0
player2_win = 0
for line in lines:
    game = line.split()
    hand1 = game[:5]
    hand2 = game[5:]
    if show_down(hand1, hand2) > 0:
        player1_win += 1
    elif show_down(hand1, hand2) < 0:
        player2_win += 1

print(player1_win, player2_win)

def show_down(hand1, hand2):
    hand_power1 = get_hand_power(hand1)
    hand_power2 = get_hand_power(hand2)
    if hand_power1[0] > hand_power2[0]:
        return 1
    elif hand_power1[0] < hand_power2[0]:
        return -1
    else:
        if hand_power1[0] >= 4:
            return compare(hand_power1[1], hand_power2[1])
        elif hand_power1[0] == 3:    # two pair
            pair1 = hand_power1[1][0]
            pair2 = hand_power2[1][0]
            compare_pair = compare(pair1, pair2)
            if compare_pair != 0:
                return compare_pair
            else:
                return compare(hand_power1[1][1], hand_power2[1][1])
        elif hand_power1[0] == 2:   # one pair
            compare_pair = compare(hand_power1[1][0], hand_power2[1][0])
            if compare_pair != 0:
                return compare_pair
            else:
                return compare(hand_power1[1][1], hand_power2[1][1])
        else:
            return compare(hand_power1[1], hand_power2[1])


def compare(rank1, rank2):
    if isinstance(rank1, list):
        for i in range(len(rank1)):
            if rank1[i] > rank2[i]:
                return 1
            elif rank1[i] < rank2[i]:
                return -1
        return 0
    else:
        return rank1 - rank2

compare(8, 8)

isinstance(4, list)