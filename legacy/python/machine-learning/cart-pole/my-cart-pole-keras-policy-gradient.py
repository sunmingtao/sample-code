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

learning_rate=0.01
done_reward = 0
n_max_episodes = 10000
n_max_steps = 10000
gamma = 0.95
epsilon = 0.0001


env = gym.make("CartPole-v0")
env = env.unwrapped
env.seed(1)

n_actions = env.action_space.n

env.reset()

print('num actions={}'.format(n_actions))

def my_loss(arg):
    action_pred, action_true, discount_episode_reward = arg
    action_true = K.cast(action_true, dtype=tf.int32)
    loss = K.sparse_categorical_crossentropy(action_true, action_pred)
    loss = loss * K.flatten(discount_episode_reward)
    return loss

def dense_model():
    state_input = Input(shape=[4], name='state_input')
    action_input = Input(shape=[1], name='action_input')
    discount_episode_reward_input = Input(shape=[1], name='reward_input')
    dense_layer1 = Dense(10, activation='relu', kernel_initializer='glorot_uniform', name='dense1')
    dense_output1 = dense_layer1(state_input)
    dense_layer2 = Dense(2, activation='relu', kernel_initializer='glorot_uniform', name='dense2')
    dense_output2 = dense_layer2(dense_output1)
    dense_layer3 = Dense(n_actions, activation='softmax', kernel_initializer='glorot_uniform', name='dense3')
    dense_output3 = dense_layer3(dense_output2)
    output_layer = Lambda(my_loss, name='output')
    loss_output = output_layer([dense_output3, action_input, discount_episode_reward_input])
    return Model(inputs=[state_input, action_input, discount_episode_reward_input], outputs=[loss_output, dense_output3])


model = dense_model()
print(model.summary())
losses = [
    lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
    lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
]
model.compile(loss=losses, optimizer=Adam(lr=learning_rate))


def preprocess(observation):
    return observation.reshape(-1, 4)


def dummy_action_input():
    return np.array([0]).reshape(-1, 1)


def dummy_reward_input():
    return np.array([100]).reshape(-1, 1)

def dummy_action_output(batch_size):
    return np.zeros(shape=(batch_size, 2))

def dummy_loss_output(batch_size):
    return np.zeros(shape=(batch_size, 1))


def discount_reward(rewards):
    discount_rewards = np.zeros_like(rewards, dtype=np.float32)
    cumulative = 0.
    for i in reversed(range(len(rewards))):
        cumulative = cumulative * gamma + rewards[i]
        discount_rewards[i] = cumulative
    discount_rewards_mean = np.mean(discount_rewards)
    discount_rewards_std = np.std(discount_rewards)
    discount_rewards = (discount_rewards - discount_rewards_mean) / discount_rewards_std
    return discount_rewards


test_rewards = [100, 200, 300]
discount_test_rewards = discount_reward(test_rewards)
assert discount_test_rewards[2] - 300 < epsilon
assert discount_test_rewards[1] - (200 + 0.99 * 300) < epsilon
assert discount_test_rewards[0] - (100 + 0.99 * (200 + 0.99 * 300)) < epsilon


def last_n_reward_average(n, game_rewards):
    last_n_rewards = game_rewards[-n:]
    return sum(last_n_rewards) / len(last_n_rewards)


best_reward = -math.inf
episode_rewards = []
average_episode_rewards = []
total_step = 0
start_time = time.time()
for episode in range(n_max_episodes):
    rewards = []
    states = []
    actions = []
    observation = env.reset()
    state = preprocess(observation)
    for step in range(n_max_steps):
        _, action_output = model.predict([state, dummy_action_input(), dummy_reward_input()])
        action = np.random.choice(np.array(range(n_actions)), size=1, p=action_output.ravel())
        observation, reward, done, _ = env.step(action[0])
        #if done:
        #    reward = done_reward
        new_state = preprocess(observation)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        state = new_state
        if done:
            break
    episode_reward = sum(rewards)
    episode_rewards.append(episode_reward)
    states = np.array(states).reshape(-1, 4)
    discount_rewards = discount_reward(rewards).reshape(-1, 1)
    actions = np.array(actions).reshape(-1, 1)
    X = [states, actions, discount_rewards]
    batch_size = len(states)
    assert batch_size == len(actions)
    assert batch_size == len(discount_rewards)
    y = [dummy_loss_output(batch_size), dummy_action_output(batch_size)]
    model.fit(X, y, batch_size=batch_size, verbose=0)
    if episode_reward > best_reward:
        best_reward = episode_reward
    avg = last_n_reward_average(100, episode_rewards)
    average_episode_rewards.append(avg)
    print('Episode {}, episode reward={}, Last 100 episode average reward={}, best reward={}'.format(episode, episode_reward, avg, best_reward))

print('Best reward is {}'.format(best_reward))
print('Training lasted {}'.format(time.time() - start_time))
plt.plot(average_episode_rewards)
plt.title('Learning rate = {}, done reward = {}'.format(learning_rate, done_reward))
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.show()