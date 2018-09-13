## 2018.9.12 

### General observation

Max number of steps of an episode is 200 in the simulation environment. Random strategy has a result of -762.


### my-taxi-keras-deep-q Run 1 

Model: Input 500 -> Dense 128 -> Dense 32 -> Dense 6\
learning_rate = 0.001\
gamma = 0.99\
n_max_episodes = 100000\
n_max_steps = 100\
loss='mse'

No memory, super naive algorithm

Observation:

After 1000 episodes, the reward seems to be stuck at -100

Episode 1254 Reward=-100 Epsilon=0.0119\
Episode 1255 Reward=-109 Epsilon=0.0119\
Episode 1256 Reward=-109 Epsilon=0.0119\
Episode 1257 Reward=-100 Epsilon=0.0118\
Episode 1258 Reward=-100 Epsilon=0.0118\
Episode 1259 Reward=-109 Epsilon=0.0118\
Episode 1260 Reward=-100 Epsilon=0.0118

### my-taxi-keras-deep-q Run 2 

Changes:\
Change loss function to tf.losses.huber_loss

Observation:\
Same as Run 1

Episode 994 Reward=-100 Epsilon=0.0169\
Episode 995 Reward=-109 Epsilon=0.0168\
Episode 996 Reward=-100 Epsilon=0.0168\
Episode 997 Reward=-109 Epsilon=0.0168\
Episode 998 Reward=-109 Epsilon=0.0167\
Episode 999 Reward=-100 Epsilon=0.0167\

I realise the agent never breaks -100 because the max steps number is set to 100 and the agent never learns to drop the passenger to the destination. All it learns is don't dropup and pickup the passenger at the wrong location

### my-taxi-keras-deep-q Run 3

Changes:\
Change n_max_steps = 150

Observation:

As expected, the reward stucks at -150

Episode 991 Reward=-159 Epsilon=0.0170\
Episode 992 Reward=-150 Epsilon=0.0169\
Episode 993 Reward=-150 Epsilon=0.0169\
Episode 994 Reward=-150 Epsilon=0.0169\
Episode 995 Reward=-159 Epsilon=0.0168\
Episode 996 Reward=-150 Epsilon=0.0168\
Episode 997 Reward=-159 Epsilon=0.0168\
Episode 998 Reward=-159 Epsilon=0.0167\
Episode 999 Reward=-150 Epsilon=0.0167

## 2018.9.12 

### my-taxi-keras-deep-q Run 4

Changes:\
Change n_max_steps = 200

Observation:

Once again, the agent can't learn much beyond not picking up/dropping off passenger illegally. The best reward obtained was -57 at the exploration period. 

Episode 232 Reward=-57 Epsilon=0.3204\
...\
Episode 992 Reward=-218 Epsilon=0.0169\
Episode 993 Reward=-209 Epsilon=0.0169\
Episode 994 Reward=-227 Epsilon=0.0169\
Episode 995 Reward=-209 Epsilon=0.0168\
Episode 996 Reward=-200 Epsilon=0.0168\
Episode 997 Reward=-209 Epsilon=0.0168\
Episode 998 Reward=-227 Epsilon=0.0167\
Episode 999 Reward=-200 Epsilon=0.0167\
Best reward is -57
