import numpy as np
import gym
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import math
import time
import matplotlib.pyplot as plt
import tensorflow as tf

actor_learning_rate=0.001
critic_learning_rate=0.005
done_reward = -10
n_max_episodes = 10000
n_max_steps = 10000
gamma = 0.99

env = gym.make("CartPole-v0")
env = env.unwrapped
env.seed(1)



n_actions = env.action_space.n

def preprocess(observation):
    return observation.reshape(-1, 4)

def get_actor_model():
    state_input = Input(shape=[4], name='state_input')
    dense_layer1 = Dense(100, activation='relu', kernel_initializer='he_uniform', name='dense1')
    dense_output1 = dense_layer1(state_input)
    dense_layer2 = Dense(50, activation='relu', kernel_initializer='he_uniform', name='dense2')
    dense_output2 = dense_layer2(dense_output1)
    dense_layer3 = Dense(n_actions, activation='softmax', kernel_initializer='he_uniform', name='dense3')
    dense_output3 = dense_layer3(dense_output1)
    return Model(inputs=state_input, outputs=dense_output3)

def get_critic_model():
    state_input = Input(shape=[4], name='state_input')
    dense_layer1 = Dense(100, activation='relu', kernel_initializer='he_uniform', name='dense1')
    dense_output1 = dense_layer1(state_input)
    dense_layer2 = Dense(50, activation='relu', kernel_initializer='he_uniform', name='dense2')
    dense_output2 = dense_layer2(dense_output1)
    dense_layer3 = Dense(1, kernel_initializer='he_uniform', name='dense3')
    dense_output3 = dense_layer3(dense_output1)
    return Model(inputs=state_input, outputs=dense_output3)

def last_n_reward_average(n, game_rewards):
    last_n_rewards = game_rewards[-n:]
    return sum(last_n_rewards) / len(last_n_rewards)


actor_model = get_actor_model()
print(actor_model.summary())
critic_model = get_critic_model()
print(critic_model.summary())

actor_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=actor_learning_rate))
critic_model.compile(loss='mse', optimizer=Adam(lr=critic_learning_rate))


def train(state, action, reward, done, new_state):
    advantages = np.zeros(shape=(1, n_actions))
    target = np.zeros(shape=(1, 1))

    current_value = critic_model.predict(state, batch_size=1)
    next_value = critic_model.predict(new_state, batch_size=1)

    if done:
        advantages[0][action] = reward - current_value
        target[0][0] = reward
    else:
        advantages[0][action] = reward + gamma * next_value - current_value
        target[0][0] = reward + gamma * next_value

    actor_model.fit(x=state, y=advantages, batch_size=1, verbose=0)
    critic_model.fit(x=state, y=target, batch_size=1, verbose=0)


best_reward = -math.inf
episode_rewards = []
average_episode_rewards = []
total_step = 0
start_time = time.time()
for episode in range(n_max_episodes):
    rewards = []
    observation = env.reset()
    state = preprocess(observation)
    for step in range(n_max_steps):
        action_output = actor_model.predict(state)
        action = np.random.choice(np.array(range(n_actions)), size=1, p=action_output.ravel())
        observation, reward, done, _ = env.step(action[0])
        if done:
            reward = done_reward
        new_state = preprocess(observation)
        rewards.append(reward)
        train(state, action, reward, done, new_state)
        state = new_state
        if done:
            break
    episode_reward = sum(rewards) - done_reward
    episode_rewards.append(episode_reward)
    if episode_reward > best_reward:
        best_reward = episode_reward
    avg = last_n_reward_average(100, episode_rewards)
    average_episode_rewards.append(avg)
    print('Episode {}, episode reward={}, Last 100 episode average reward={}, best reward={}'.format(episode, episode_reward, avg, best_reward))


import tensorflow as tf
import keras.backend as K
from keras.losses import mean_squared_error


def my_loss(y_pred, y_true):
    return mean_squared_error(y_true, y_pred)


y_pred = tf.Variable([1.5])
y_true = tf.Variable([[2.0]])

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

loss = my_loss(y_pred, y_true)

loss_val = sess.run([loss])
print (loss_val)