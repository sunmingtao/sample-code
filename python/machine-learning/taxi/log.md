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

## 2018.9.13 

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


![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-4.png)

### my-taxi-keras-deep-q-memory Run 5

Use dequeue memory to store the experience, and train the experience in batch

memory_capacity = 10000\
BATCH_SIZE = 32\
n_warm_up_episode = 10\

Observation:

Achieved some good results after 1000 episodes of training

Episode 990 Reward=14 Epsilon=0.0170\
Episode 991 Reward=7 Epsilon=0.0170\
Episode 992 Reward=8 Epsilon=0.0169\
Episode 993 Reward=8 Epsilon=0.0169\
Episode 994 Reward=11 Epsilon=0.0169\
Episode 995 Reward=4 Epsilon=0.0168\
Episode 996 Reward=8 Epsilon=0.0168\
Episode 997 Reward=14 Epsilon=0.0168\
Episode 998 Reward=9 Epsilon=0.0167\
Episode 999 Reward=9 Epsilon=0.0167\
Best reward is 15

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-5.png)

### my-taxi-keras-deep-q-memory-fixed-target Run 6

Use another target model to calculate the new state's q value

n_target_model_update_every_steps=1000

Observation:

Achieved similar result as run 5 but with larger variance. 

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-6.png)

### my-taxi-keras-deep-q-memory-fixed-target Run 7

Changes:\
n_target_model_update_every_steps=5000

Observation:

It's worse than run 6. Obviously target model should be updated more frequently. 

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-7.png)


### my-taxi-keras-deep-q-memory-fixed-target Run 8

Changes:\
n_target_model_update_every_steps=300

Observation:

Better than Run 7. Close to Run 6. Doesn't seem to help reduce the variance (Compared to Run 5)

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-8.png)

### my-taxi-keras-deep-q-memory Run 9

Based on Run 5, train 2000 episode

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-9.png)

### my-taxi-keras-deep-q-memory-fixed-target Run 10

Based on Run 8, train 2000 episode

Observation:

Fixed target seems to be slightly better than no fixed target

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taix-10.png)

## 2018.9.14

### my-taxi-keras-deep-q-memory Run 11

Based on Run 9, change learning rate from 0.001 to 0.01

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-11.png)

### my-taxi-keras-deep-q-memory-fixed-target Run 12

Based on Run 10, change learning rate from 0.001 to 0.01

Oberservation:

The result is abysmal. Even worse than the no memory version.  

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-12.png)

### my-taxi-keras-deep-q-memory Run 13

Based on Run 11, change learning rate from 0.01 to 0.005

![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-13.png)

### my-taxi-keras-deep-q-memory-fixed-target Run 14

Based on Run 12, change learning rate from 0.01 to 0.005

Oberservation:


![alt text](https://github.com/sunmingtao/sample-code/blob/master/python/machine-learning/taxi/taxi-14.png)
