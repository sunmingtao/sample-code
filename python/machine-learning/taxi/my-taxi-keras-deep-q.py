import numpy as np
import gym
import random
from keras.utils.np_utils import to_categorical

env = gym.make("Taxi-v2")

n_actions = env.action_space.n
n_states = env.observation_space.n

print('num actions={}, num states={}'.format(n_actions, n_states))

def preprocess(observation):
    return to_categorical(observation, num_classes=n_states)



observation = env.reset()
state = preprocess(observation)

env.render()
for action in range(6):
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    env.render()

env.step(0)
env.step(0)
env.render()
env.step(4)
env.render()
'''
- 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
'''
