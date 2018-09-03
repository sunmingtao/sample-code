import numpy as np
from math import nan


# Know transition probability and reward

ts0 = np.array([[0.7, 0.3, 0], [1, 0, 0], [0.8, 0.2, 0]])
ts1 = np.array([[0, 1, 0], [nan, nan, nan], [0, 0, 1]])
ts2 = np.array([[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]])
T = np.array([ts0, ts1, ts2]) # (current_state, action, next_state)


rs0 = np.array([[10, 0, 0], [0, 0, 0], [0, 0, 0]])
rs1 = np.array([[0, 0, 0], [nan, nan, nan], [0, 0, -50]])
rs2 = np.array([[nan, nan, nan], [40, 0, 0], [nan, nan, nan]])
R = np.array([rs0, rs1, rs2]) # (current_state, action, next_state)

possible_actions = [[0, 1, 2], [0,2], [1]]

Q = np.full((3, 3), -np.inf) # (state, action)

for index, actions in enumerate(possible_actions):
    Q[index, actions] = 0



discount_rate = 0.95
n_iterations = 20000


for i in range(n_iterations):
    Q_prev = Q.copy()
    for state in range(3):
        for action in possible_actions[state]:
            Q[state, action] = np.sum(T[state, action, next_state] * (R[state, action, next_state] + discount_rate * np.max(Q_prev[next_state])) for next_state in range(3))

print(Q)

# Doesn't know transition probability and reward

learning_rate0 = 1
learning_rate_decay = 0.1
discount_rate = 0.95
n_iterations = 20000

s = 0

Q = np.full((3,3), -np.inf)

for index, actions in enumerate(possible_actions):
    Q[index, actions] = 0

for iteration in range(n_iterations):
    a = np.random.choice(possible_actions[s])
    sp = np.random.choice(range(3), p=T[s, a])
    reward = R[s, a, sp]
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    Q[s, a] = (1 - learning_rate) * Q[s,a] + learning_rate * (reward + discount_rate * np.max(Q[sp]))
    s = sp

print(Q)