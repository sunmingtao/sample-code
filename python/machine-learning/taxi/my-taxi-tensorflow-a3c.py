import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import gym
import threading

from tensorflow.python import keras
from tensorflow.python.keras import layers

env = gym.make("Taxi-v2")

n_actions = 6
n_states = 500
n_max_episode=1200
gamma=0.99
update_freq=20
learning_rate=0.01


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

def preprocess(observation):
    return tf.one_hot(observation, depth=n_states).numpy()[None, :]


class ActorCriticModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.dense_layer1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense1')
        self.policy_logits_layer = layers.Dense(n_actions, kernel_initializer='he_uniform', name='policy_logits')
        self.dense_layer2 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense2')
        self.state_values_layer = layers.Dense(1, kernel_initializer='he_uniform', name='state_values')

    def call(self, inputs):
        dense_output1 = self.dense_layer1(inputs)
        policy_logits = self.policy_logits_layer(dense_output1)
        dense_output2 = self.dense_layer2(inputs)
        state_values = self.state_values_layer(dense_output2)
        return policy_logits, state_values


global_model = ActorCriticModel()
obs = env.reset()
state = preprocess(obs)
global_model(tf.convert_to_tensor(state))
#print(global_model.trainable_weights)

class Worker(threading.Thread):
    global_episode = 0
    global_moving_average_reward = 0


    def __init__(self, worker_id):
        super().__init__()
        self.worker_id = worker_id
        self.env = gym.make("Taxi-v2")
        self.local_model = ActorCriticModel()
        self.optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        self.memory = Memory()

    def run(self):
        print('Worker {} Run'.format(self.worker_id))
        time_count = 1
        while Worker.global_episode < n_max_episode:
            self.memory.clear()
            episode_reward = 0
            observation = self.env.reset()
            state = preprocess(observation)
            done = False
            while not done:
                policy_logits, state_values = self.local_model(tf.convert_to_tensor(state))
                policy_probability = tf.nn.softmax(policy_logits)
                print(policy_probability)
                action = np.random.choice(n_actions, p=policy_probability.numpy()[0])
                new_observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.memory.store(state, action, reward)
                new_state = preprocess(new_observation)
                if time_count % update_freq == 0 or done:
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(new_state, done)
                    grads = tape.gradient(total_loss, self.local_model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, global_model.trainable_weights))
                    self.local_model.set_weights(global_model.get_weights())
                    self.memory.clear()
                    if done:
                        if Worker.global_moving_average_reward == 0:
                            Worker.global_moving_average_reward = episode_reward
                        else:
                            Worker.global_moving_average_reward = Worker.global_moving_average_reward * 0.99 + episode_reward * 0.01
                time_count += 1
                state = new_state
            Worker.global_episode += 1
            print('Global episode {}, episode reward {}, global moving average {}'.format(Worker.global_episode, episode_reward, Worker.global_moving_average_reward))


    def compute_loss(self, new_state, done):
        if done:
            reward_sum = 0
        else:
            reward_sum = self.local_model(tf.convert_to_tensor(new_state))[1].numpy()[0][0]
        discounted_rewards = []
        for reward in self.memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        policy_logits, state_values = self.local_model(tf.convert_to_tensor(np.vstack(self.memory.states), dtype=tf.float32))
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - state_values
        state_value_loss = advantage ** 2
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.memory.actions, logits=policy_logits)
        advantage = tf.stop_gradient(advantage)
        policy_loss *= advantage
        policy_softmax = tf.nn.softmax(policy_logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy_softmax, logits=policy_logits)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * state_value_loss + policy_loss))
        return total_loss

worker = Worker(1)
worker.start()



