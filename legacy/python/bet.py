import numpy as np

n_games = 500000
initial_bankroll = 100


# Strategy 1 - bet 1% of the bankroll always on 6

bet_amount_proportion = 0.006

distribution = np.array([0.166, 0.166, 0.166, 0.166, 0.166, 0.17])

bankrolls = []
for i in range(100):
    bankroll = initial_bankroll
    outcomes = np.random.choice(np.array(range(1, 7)), size=n_games, p=distribution)
    for outcome in outcomes:
        bet_amount = bankroll * bet_amount_proportion
        bankroll -= bet_amount
        if outcome == 6:
            bankroll += bet_amount * 6
    bankrolls.append(bankroll)

n_wins = len([b for b in bankrolls if b > initial_bankroll])
n_losses = 100 - n_wins
print ('avg bankrolls = {}, min = {}, max = {}, wins ={}, losses = {}'.format(sum(bankrolls)/len(bankrolls), min(bankrolls), max(bankrolls), n_wins, n_losses))


# Strategy 2 - bet on all numbers with amount proportional to their distribution

bankrolls = []
for i in range(100):
    bankroll = initial_bankroll
    outcomes = np.random.choice(np.array(range(1, 7)), size=n_games, p=distribution)
    for outcome in outcomes:
        bankroll = distribution[outcome-1] * bankroll * 6
    bankrolls.append(bankroll)
    print(bankroll)

n_wins = len([b for b in bankrolls if b > initial_bankroll])
n_losses = 100 - n_wins
print ('avg bankrolls = {}, min = {}, max = {}, wins ={}, losses = {}'.format(sum(bankrolls)/len(bankrolls), min(bankrolls), max(bankrolls), n_wins, n_losses))