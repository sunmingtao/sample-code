'''Huber loss
after 4000 games training, average score fluctuates around 400
'''
import gym
import numpy as np
import tensorflow as tf
import os


import sys

print(os.listdir('.'))

env = gym.make("MsPacman-v0")

mspacman_color = 210 + 164 + 74
n_outputs = env.action_space.n
max_memory_capacity = 10000
batch_size = 32
n_epoches = 5000
n_max_steps = 10000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
discount_rate = 0.95
input_height = 88
input_width = 80
input_channels = 1
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
iteration = 0
skip_start = 90  # Skip the start of every game (it's just waiting time).
checkpoint_path = "./my_pacman-tensorflow-003.ckpt"

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)



class Memory:
    def __init__(self, capacity=max_memory_capacity):
        self.memory = np.empty(shape=max_memory_capacity, dtype=np.object)
        self.index = 0
        self.length = 0
        self.capacity = capacity

    def append(self, data):
        self.memory[self.index] = data
        self.length = min(self.length+1, self.capacity)
        self.index += 1
        if self.index >= self.capacity:
            self.index = 0

    def sample(self, batch_size=batch_size):
        indexes = np.random.permutation(self.length)[:batch_size]
        return self.memory[indexes]


def q_network(state_tensor):
    inputs = state_tensor
    conv_outputs1 = tf.layers.conv2d(inputs, filters=32, kernel_size=(8,8), strides=4, padding='same', activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
    conv_outputs2 = tf.layers.conv2d(conv_outputs1, filters=64, kernel_size=(4,4), strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
    conv_outputs3 = tf.layers.conv2d(conv_outputs2, filters=64, kernel_size=(3,3), strides=1, padding='same', activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
    flat_outputs = tf.reshape(conv_outputs3, shape=[-1, n_hidden_in])
    dense_outputs = tf.layers.dense(flat_outputs, n_hidden, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
    outputs = tf.layers.dense(dense_outputs, n_outputs, kernel_initializer=tf.variance_scaling_initializer())
    return outputs


tf.reset_default_graph()

state_tensor = tf.placeholder(tf.float32, shape=(None, input_height, input_width, input_channels))
q_tensor = q_network(state_tensor)
labels_tensor = tf.placeholder(dtype=tf.float32, shape=(None, n_outputs))
loss = tf.losses.huber_loss(labels=labels_tensor, predictions=q_tensor)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)
saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


def reshape_state(state_val):
    return state_val.reshape(1, input_height, input_width, input_channels)

def predict(state_val):
    return sess.run(q_tensor, feed_dict={state_tensor : reshape_state(state_val)})


def act(state_val):
    if np.random.rand() <= epsilon:
        return np.random.randint(n_outputs)
    else:
        predicted_q = predict(state_val)
        return np.argmax(predicted_q, axis=-1)[0]


def train(training_data):
    global epsilon
    states = np.array([data[0] for data in training_data])
    actions = np.array([data[1] for data in training_data])
    rewards = np.array([data[2] for data in training_data])
    next_states = np.array([data[3] for data in training_data])
    dones = np.array([data[4] for data in training_data])
    next_state_q_vals = sess.run(q_tensor, feed_dict={state_tensor: next_states})
    label_q_vals = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * discount_rate * np.amax(next_state_q_vals, axis=-1).reshape(-1, 1) # shape = (batch_size, 1)
    state_q_vals = sess.run(q_tensor, feed_dict={state_tensor: states}) # shape = (batch_size, 2)
    for state_q_val, action, label_q_val, done in zip(state_q_vals, actions, label_q_vals, dones):
        if done:
            state_q_val[action] = -500
        else:
            state_q_val[action] = label_q_val[0]
    sess.run(training_op, feed_dict={state_tensor: states, labels_tensor: state_q_vals})
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


def last_n_reward_average(n, game_rewards):
    last_n_rewards = game_rewards[-n:]
    return sum(last_n_rewards) / len(last_n_rewards)


memory = Memory()

game_rewards = []
max_reward = 0
for game in range(n_epoches):
    state = env.reset()
    for skip in range(skip_start):  # skip the start of each game
        state, _, _, _ = env.step(0)
    state = preprocess_observation(state)
    total_reward = 0
    for step in range(n_max_steps):
        #env.render()
        iteration += 1
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_observation(next_state)
        total_reward += reward
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print('Game {}, reward {}, Epsilon {:.2}'.format(game, total_reward, epsilon))
            break
        if iteration >= 50000 and iteration % 4 == 0:
            data = memory.sample()
            train(data)
    game_rewards.append(total_reward)
    if total_reward > max_reward:
        max_reward = total_reward
    if (game + 1) % 10 == 0:  # Save every 10 games
        saver.save(sess, checkpoint_path)
    print('Last 30 games average reward is {:.4}. Max reward is {}'.format(last_n_reward_average(30, game_rewards), max_reward))
