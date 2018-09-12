## 2018.9.12 

Run my-taxi-keras-deep-q

Model: Input 500 -> Dense 128 -> Dense 32 -> Dense 6
learning_rate = 0.001
gamma = 0.99
n_max_episodes = 100000
n_max_steps = 100

No memory, super naive algorithm

Observation:

After 1000 episodes, the reward seems to be stuck at -100

Episode 1254 Reward=-100 Epsilon=0.0119
Episode 1255 Reward=-109 Epsilon=0.0119
Episode 1256 Reward=-109 Epsilon=0.0119
Episode 1257 Reward=-100 Epsilon=0.0118
Episode 1258 Reward=-100 Epsilon=0.0118
Episode 1259 Reward=-109 Epsilon=0.0118
Episode 1260 Reward=-100 Epsilon=0.0118
