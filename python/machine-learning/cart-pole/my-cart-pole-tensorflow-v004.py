import gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os



env = gym.make("CartPole-v1")

n_epoches = 1000
n_outputs = env.action_space.n
epsilon = 1.0
epsilon_min = 0.01
n_max_steps = 500
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
optimizer = tf.train.AdamOptimizer()
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
    for state, action, reward, next_state, done in training_data:
        next_state_q_val = session.run(q_tensor, feed_dict={state_tensor: next_state.reshape(1,4)})
        label_q_val = reward
        if not done:
            label_q_val += discount_rate * np.amax(next_state_q_val)
        state_q_val = session.run(q_tensor, feed_dict={state_tensor: state.reshape(1,4)})
        state_q_val[0][action] = label_q_val
        _, loss_val = session.run([training_op, loss], feed_dict={state_tensor: state.reshape(1,4), labels_tensor: state_q_val})
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


memory = Memory()

game_steps = []
max_step = 0
for game in range(n_epoches):
    state = env.reset()
    total_reward = 0
    for step in range(n_max_steps):
        env.render()
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        #reward = reward if not done else -10
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
    print('Avg game step is {:.4}. Max step is {}'.format(sum(game_steps) / len(game_steps), max_step))




