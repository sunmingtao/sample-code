'''based on my-taxi-keras-dueling-dqn-v3, implement priority dqn'''

import numpy as np
import gym
import random
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import math
from collections import deque
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import operator

learning_rate = 0.001
gamma = 0.99
n_max_episodes = 2000
n_max_steps = 200
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
memory_capacity = 200000
BATCH_SIZE = 32
n_warm_up_episode = 50
n_target_model_update_every_steps=300
alpha = 0.6


env = gym.make("Taxi-v2")

n_actions = env.action_space.n
n_states = env.observation_space.n

print('num actions={}, num states={}'.format(n_actions, n_states))

class SegmentTree():
    def __init__(self, capacity, operation, init_value=None):
        # must be power of 2
        self.capacity = self.__ceiling_power_2(capacity)
        self._value = [init_value for _ in range(2 * self.capacity)]
        self.operation = operation

    def __ceiling_power_2(self, capacity):
        temp = 1
        while temp < capacity:
            temp *= 2
        return temp

    def __setitem__(self, key, value):
        temp_index = key+self.capacity
        self._value[temp_index] = value
        while temp_index > 1:
            temp_index //= 2
            self._value[temp_index] = self.operation(self._value[temp_index * 2], self._value[temp_index * 2 + 1])

    def __getitem__(self, item):
        return self._value[item+self.capacity]

    def __op(self, start, end, lower, upper, sum_index):
        if start == lower and end == upper:
            return self._value[sum_index]
        else:
            mid = (lower + upper) // 2
            if mid < start:
                return self.__op(start, end, mid+1, upper, sum_index * 2 + 1)
            elif mid >= end:
                return self.__op(start, end, lower, mid, sum_index * 2)
            else:
                return self.__op(start, mid, lower, mid, sum_index * 2) + self.__op(mid + 1, end, mid + 1, upper, sum_index * 2 + 1)

    def op(self, start, end):
        lower = 0
        upper = self.capacity - 1
        sum_index = 1
        return self.__op(start, end, lower, upper, sum_index)

    def op_all(self):
        return self._value[1]


class SumTree(SegmentTree):

    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=operator.add, init_value=0)

    def sum(self, start, end):
        return super().op(start, end)


    def sum_all(self):
        return super().op_all()


    '''Find the highest index 'i' in the array such that sum(arr[0] + arr[1] + ... + arr[i - i]) <= upper'''
    def find_propotional_index(self, random_sum):
        assert 0 <= random_sum < self.sum_all()
        upper = random_sum
        index = 1
        while index < self.capacity:
            if self._value[index * 2] < upper:
                upper -= self._value[index * 2]
                index = index * 2 + 1
            else:
                index *= 2
        return index-self.capacity



test_sum_tree = SumTree(7)
assert test_sum_tree.capacity == 8
for i in range(8):
    test_sum_tree[i]= i+1
assert test_sum_tree.sum_all() == 36
assert test_sum_tree.sum(0, 7) == 36
assert test_sum_tree.sum(0, 0) == 1
assert test_sum_tree.sum(1, 1) == 2
assert test_sum_tree.sum(0, 3) == 10
assert test_sum_tree.sum(0, 4) == 15
assert test_sum_tree.sum(2, 4) == 12
assert test_sum_tree.sum(3, 5) == 15
assert test_sum_tree.sum(3, 7) == 30

assert test_sum_tree.find_propotional_index(9.9) == 3
assert test_sum_tree.find_propotional_index(10) == 3
assert test_sum_tree.find_propotional_index(10.5) == 4

class MinTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity=capacity, operation=min, init_value=float('inf'))

    def min(self, start, end):
        return super().op(start, end)

    def min_all(self):
        return super().op_all()


test_min_tree = MinTree(7)
for i in range(8):
    test_min_tree[i]= i+1
assert test_min_tree.min_all() == 1
assert test_min_tree.min(0, 7) == 1
assert test_min_tree.min(0, 3) == 1
assert test_min_tree.min(4, 7) == 5



class Memory:


    def __init__(self, capacity=memory_capacity):
        self.memory = np.empty(shape=capacity, dtype=np.object)
        self.capacity = capacity
        self.current_index = 0
        self.size = 0
        self.sum_tree = SumTree(capacity=capacity)
        self.min_tree = MinTree(capacity=capacity)
        self.max_priority = 1.
        self.start_beta = 0.04
        self.end_beta = 1

    def append(self, state, action, reward, done, new_state):
        self.memory[self.current_index] = (state, action, reward, done, new_state)
        self.sum_tree[self.current_index] = self.max_priority ** alpha
        self.min_tree[self.current_index] = self.max_priority ** alpha
        self.size = min(self.size + 1, self.capacity)
        self.current_index = (self.current_index + 1) % self.capacity

    def update_priority(self, indexes, priorities):
        for index, priority in zip(indexes, priorities):
            priority = priority ** alpha
            self.sum_tree[index] = priority
            self.min_tree[index] = priority
            self.max_priority = max(self.max_priority, priority)

    def calculate_beta(self, current_episode):
        a = (self.end_beta - self.start_beta) / n_max_episodes
        b = self.start_beta
        return min(self.end_beta, a * current_episode + b)

    def sample(self, beta=0.6, batch_size=BATCH_SIZE):
        '''
        indexes = []
        tree_sum = self.sum_tree.sum_all()
        for _ in range(batch_size):
            indexes.append(self.sum_tree.find_propotional_index(random.random() * tree_sum))
        return self.bulk_get(indexes, beta)
        '''
        indexes = np.random.permutation(self.size)[:batch_size]
        return self.memory[indexes], indexes, np.ones(shape=(batch_size, 1))

    def bulk_get(self, indexes, beta):
        tree_sum = self.sum_tree.sum_all()
        experiences = []
        importance_weights = []
        min_p = self.min_tree.min_all() / tree_sum
        max_importance_weight = (min_p * self.size) ** (-beta)
        for index in indexes:
            experiences.append(self.memory[index])
            pi = self.sum_tree[index] / tree_sum
            importance_weight_i = (pi * self.size) ** (-beta)
            importance_weights.append(importance_weight_i / max_importance_weight)
        return experiences, indexes, importance_weights


test_memory = Memory(3)
test_memory.append(1,2,3,4,5)
assert test_memory.current_index == 1
assert test_memory.size == 1
test_memory.append(1,2,3,4,5)
assert test_memory.current_index == 2
assert test_memory.size == 2
test_memory.append(1,2,3,4,5)
assert test_memory.current_index == 0
assert test_memory.size == 3
test_memory.append(1,2,3,4,5)
assert test_memory.current_index == 1
assert test_memory.size == 3

test_memory = Memory(100)
test_memory.append(1,2,3,4,5)
test_memory.append(1,2,3,4,5)
experiences, indexes, importance_weights = test_memory.sample(batch_size=10)
for e, i, iw in zip(experiences, indexes, importance_weights):
    assert 0 <= i < test_memory.size

test_memory = Memory(100)
test_memory.append(1,2,3,4,5)
test_memory.append(1,2,3,4,5)
test_memory.append(1,2,3,4,5)
test_memory.update_priority([0,1,2], [0.2, 0.3, 0.5])
experiences, indexes, iws = test_memory.bulk_get([0,1,2], beta=0.6)

assert test_memory.sum_tree[0] == 0.2 ** alpha
assert test_memory.sum_tree[1] == 0.3 ** alpha
assert test_memory.sum_tree[2] == 0.5 ** alpha
assert 1 >= iws[0] > iws[1] > iws[2] > 0



memory = Memory()


def get_epsilon(episode):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


def get_action(state):
    # 3. Choose an action a in the current world state (s)
    ran = random.uniform(0, 1)
    if ran > epsilon:
        return np.argmax(q_table[observation, :])
    else:
        return env.action_space.sample()


''' Convert observation to one hot vector '''
def preprocess(observation):
    return to_categorical(observation, num_classes=n_states).reshape(-1, n_states)


def get_q_values_batch_by_actions(q_values_batch, action_batch):
    batch_size = len(q_values_batch)
    assert batch_size == len(action_batch)
    max_q_value_for_action_batch = q_values_batch[range(batch_size), action_batch].reshape(-1, 1)
    assert max_q_value_for_action_batch.shape == (batch_size, 1)
    return max_q_value_for_action_batch


def combine_state_value_and_advantage(arg):
    state_value = arg[0]
    advantage = arg[1]
    return state_value + (advantage - K.mean(advantage, keepdims=True))

def huber_loss(arg):
    y_true, y_pred, importance_weights = arg
    x = y_true - y_pred
    squared_loss = .5 * K.square(x)
    linear_loss = K.abs(x) - .5
    loss = tf.where(K.abs(x) < 1., squared_loss, linear_loss)
    loss = tf.multiply(loss, importance_weights)
    return K.sum(loss, axis=-1)


def dqn_model():
    inputs = Input(shape=[n_states], name='input')
    dense_layer1 = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense1')
    dense_outputs1 = dense_layer1(inputs)
    dense_layer2 = Dense(32, activation='relu', kernel_initializer='he_uniform', name='dense2')
    dense_outputs2 = dense_layer2(dense_outputs1)
    advantage_layer = Dense(n_actions, activation='linear', kernel_initializer='TruncatedNormal', name='advantage')
    advantage_outputs = advantage_layer(dense_outputs2)
    state_value_layer = Dense(1, activation='linear', kernel_initializer='TruncatedNormal', name='state_value')
    state_value_outputs = state_value_layer(dense_outputs2)
    output_layer = Lambda(combine_state_value_and_advantage, name='output')
    outputs = output_layer([state_value_outputs, advantage_outputs])
    return Model(inputs=inputs, outputs=outputs)


def dqn_train_model(model):
    inputs = model.input
    y_true = Input(shape=[n_actions], name='y_true')
    importance_weights = Input(shape=[1], name='importance_weights')
    outputs = model.output
    loss_layer = Lambda(huber_loss, name='loss')
    loss_outputs = loss_layer([y_true, outputs, importance_weights])
    return Model(inputs=[inputs, y_true, importance_weights], outputs=loss_outputs)


model = dqn_model()
target_model = dqn_model()
train_model = dqn_train_model(model)

print(model.summary())
print(train_model.summary())

observation = env.reset()
state = preprocess(observation)
q_values = model.predict(state)
assert q_values.shape == (1, 6)
action = np.argmax(q_values)
assert type(action) is np.int64 and 0 <= action <= 6
max_q_value = np.max(q_values)
assert type(max_q_value) is np.float32

model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=Adam(lr=learning_rate))


best_reward = -math.inf
episode_rewards = []
total_step = 0
start_time = time.time()
for episode in range(n_max_episodes):
    observation = env.reset()
    state = preprocess(observation)
    step = 0
    episode_reward = 0
    for step in range(n_max_steps):
        state_q_values = model.predict(state)
        ran = random.uniform(0, 1)
        if ran > epsilon:
            action = np.argmax(state_q_values)
        else:
            action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        new_state = preprocess(observation)
        memory.append(state, action, reward, done, new_state)
        if episode > n_warm_up_episode:
            current_beta = memory.calculate_beta(episode)
            experiences, indexes, importance_weights = memory.sample(beta=0)
            assert len(experiences) == BATCH_SIZE
            assert len(indexes) == BATCH_SIZE
            assert len(importance_weights) == BATCH_SIZE
            states = []
            actions = []
            rewards = []
            dones = []
            new_states = []
            for _state, _action, _reward, _done, _new_state in experiences:
                states.append(_state)
                actions.append(_action)
                rewards.append(_reward)
                dones.append(_done)
                new_states.append(_new_state)
            states_batch = np.array(states).reshape(-1, n_states)
            actions_batch = np.array(actions).reshape(-1, 1)
            rewards_batch = np.array(rewards).reshape(-1, 1)
            dones_batch = np.array(dones).reshape(-1, 1)
            new_states_batch = np.array(new_states).reshape(-1, n_states)

            new_state_q_values_batch = model.predict_on_batch(new_states_batch)
            assert new_state_q_values_batch.shape == (BATCH_SIZE, n_actions)
            max_action_new_state_q_values_batch = np.argmax(new_state_q_values_batch, axis=-1)
            assert max_action_new_state_q_values_batch.shape == (BATCH_SIZE, )
            target_new_state_q_values_batch = target_model.predict_on_batch(new_states_batch)
            assert target_new_state_q_values_batch.shape == (BATCH_SIZE, n_actions)
            q_values_select_action_batch = get_q_values_batch_by_actions(target_new_state_q_values_batch, max_action_new_state_q_values_batch)
            new_state_q_values_for_action_batch = rewards_batch + (1 - dones_batch) * gamma * q_values_select_action_batch
            assert new_state_q_values_for_action_batch.shape == (BATCH_SIZE, 1)
            state_q_values_batch = model.predict_on_batch(states_batch)
            y_pred = state_q_values_batch.copy()
            for _state_q_values, _action, _new_state_q_values_for_action in zip(state_q_values_batch, actions_batch, new_state_q_values_for_action_batch):
                _state_q_values[_action] = _new_state_q_values_for_action[0]
            dummy_target = np.zeros(shape=(BATCH_SIZE, n_actions))
            importance_weights = np.array(importance_weights).reshape(-1, 1)
            train_model.fit(x=[states_batch, state_q_values_batch, importance_weights], y=dummy_target, batch_size=BATCH_SIZE, verbose=0)
            y_true = state_q_values_batch
            new_priorities = abs(np.sum(y_pred - y_true, axis=-1)) + 0.001
            memory.update_priority(indexes, new_priorities)
        state = new_state
        total_step += 1
        if total_step % n_target_model_update_every_steps == 0:
            target_model.set_weights(model.get_weights())
        if done:
            break
    if episode_reward > best_reward:
        best_reward = episode_reward
    if episode > n_warm_up_episode:
        epsilon = get_epsilon(episode-n_warm_up_episode)
    episode_rewards.append(episode_reward)
    print('Episode {} Reward={} Epsilon={:.4f}'.format(episode, episode_reward, epsilon))
print('Best reward is {}'.format(best_reward))
print('Training lasted {}'.format(time.time() - start_time))


plt.plot(episode_rewards)
plt.title('Priority replay. n_max_steps={}'.format(n_max_steps))
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.show()

