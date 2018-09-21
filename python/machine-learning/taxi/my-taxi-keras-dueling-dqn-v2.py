'''based on my-taxi-keras-dueling-dqn, refactor get_q_values_batch_by_actions()'''

import numpy as np
import gym
import random
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from tensorflow.losses import huber_loss
import math
from collections import deque
import time
import matplotlib.pyplot as plt

learning_rate = 0.001
gamma = 0.99
n_max_episodes = 2000
n_max_steps = 200
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
memory_capacity = 10000
BATCH_SIZE = 32
n_warm_up_episode = 50
n_target_model_update_every_steps=300

env = gym.make("Taxi-v2")

n_actions = env.action_space.n
n_states = env.observation_space.n

print('num actions={}, num states={}'.format(n_actions, n_states))


class Memory:

    def __init__(self, capacity=memory_capacity):
        self.memory = deque(maxlen=capacity)

    def append(self, state, action, reward, done, new_state):
        self.memory.append((state, action, reward, done, new_state))

    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.memory, batch_size)


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



model = dqn_model()
target_model = dqn_model()


print(model.summary())

observation = env.reset()
state = preprocess(observation)
q_values = model.predict(state)
assert q_values.shape == (1, 6)
action = np.argmax(q_values)
assert type(action) is np.int64 and 0 <= action <= 6
max_q_value = np.max(q_values)
assert type(max_q_value) is np.float32

model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

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
            experiences = memory.sample()
            assert len(experiences) == BATCH_SIZE
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
            for _state_q_values, _action, _new_state_q_values_for_action in zip(state_q_values_batch, actions_batch, new_state_q_values_for_action_batch):
                _state_q_values[_action] = _new_state_q_values_for_action[0]
            model.fit(x=states_batch, y=state_q_values_batch, batch_size=BATCH_SIZE, verbose=0)
        state = new_state
        total_step += 1
        if total_step % n_target_model_update_every_steps == 0:
            target_model.set_weights(model.get_weights())
            print('Target model weights updated at step {}'.format(total_step))
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
plt.title('Dueling DQN. Tensorflow huberloss.')
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.show()


import tensorflow as tf
from keras.losses import mean_squared_error

def my_loss(y_pred, y_true, weights):
    x = y_true - y_pred
    loss = tf.square(x) * weights
    return tf.reduce_sum(loss, axis=-1)

def my_loss2(y_pred, y_true):
    x = y_true - y_pred
    loss = tf.square(x)
    return tf.reduce_sum(loss, axis=-1)


y_pred = tf.Variable([[1.1, 2.3, 3.],[1.2, 2.5, 10.]])
y_true = tf.Variable([[1., 2., 4.],[2., 2., 3.]])
weights = tf.Variable([[0.8],[0.7]])

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

loss = mean_squared_error(y_pred, y_true)
loss2 = my_loss(y_pred, y_true, weights)
loss3 = my_loss2(y_pred, y_true)

loss_val, loss2_val, loss3_val = sess.run([loss, loss2, loss3])
print (loss_val, loss2_val, loss3_val)




