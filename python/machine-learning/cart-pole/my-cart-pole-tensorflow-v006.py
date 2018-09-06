'''
Based on version 5
'''
import gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os



env = gym.make("CartPole-v0")
env._max_episode_steps = 5000


n_epoches = 5000
n_outputs = env.action_space.n
epsilon = 1.0
epsilon_min = 0.01
n_max_steps = 5000
batch_size = 32
max_memory_capacity = 2000
discount_rate = 0.95
epsilon_decay = 0.995
memory = np.empty(shape=max_memory_capacity, dtype=np.object)

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
    dense_outputs1 = tf.layers.dense(inputs=inputs, units=30, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
    dense_outputs2 = tf.layers.dense(inputs=dense_outputs1, units=30, activation=tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
    outputs = tf.layers.dense(inputs=dense_outputs2, units=n_outputs, kernel_initializer=tf.variance_scaling_initializer())
    return outputs


tf.reset_default_graph()

state_tensor = tf.placeholder(tf.float32, shape=(None, 4))
q_tensor = q_network(state_tensor)


labels_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 2))
loss = tf.reduce_mean(tf.square(labels_tensor - q_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

session = tf.InteractiveSession()
session.run(tf.initialize_all_variables())


def predict(state_val):
    return session.run(q_tensor, feed_dict={state_tensor : state_val.reshape(1,4)})


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
    done = np.array([data[4] for data in training_data])
    next_state_q_vals = session.run(q_tensor, feed_dict={state_tensor: next_states})
    label_q_vals = rewards.reshape(-1, 1) + (1 - done.reshape(-1, 1)) * discount_rate * np.amax(next_state_q_vals, axis=-1).reshape(-1, 1) # shape = (batch_size, 1)
    state_q_vals = session.run(q_tensor, feed_dict={state_tensor: states}) # shape = (batch_size, 2)
    for state_q_val, action, label_q_val in zip(state_q_vals, actions, label_q_vals):
        state_q_val[action] = label_q_val[0]
    session.run(training_op, feed_dict={state_tensor: states, labels_tensor: state_q_vals})
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


def last_50_reward_average(game_rewards):
    last_50_rewards = game_rewards[-50:]
    return sum(last_50_rewards) / len(last_50_rewards)


memory = Memory()

game_steps = []
max_step = 0
for game in range(n_epoches):
    state = env.reset()
    total_reward = 0
    for step in range(n_max_steps):
        #env.render()
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print('Game {}, Step {}, Epsilon {:.2}'.format(game, step, epsilon))
            break
        if memory.length >= batch_size:
            data = memory.sample()
            train(data)
    game_steps.append(step)
    if step > max_step:
        max_step = step
    print('Last 50 game average step is {:.4}. Max step is {}'.format(last_50_reward_average(game_steps), max_step))

