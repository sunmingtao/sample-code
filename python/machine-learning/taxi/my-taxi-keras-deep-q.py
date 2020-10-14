import numpy as np
import gym
import random
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.losses import huber_loss
import math
import matplotlib.pyplot as plt

learning_rate = 0.001
gamma = 0.99
n_max_episodes = 1000
n_max_steps = 200
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005


env = gym.make("Taxi-v2")

n_actions = env.action_space.n
n_states = env.observation_space.n

print('num actions={}, num states={}'.format(n_actions, n_states))

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


def dqn_model():
    inputs = Input(shape=[n_states], name='input')
    dense_layer1 = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense1')
    dense_outputs1 = dense_layer1(inputs)
    dense_layer2 = Dense(32, activation='relu', kernel_initializer='he_uniform', name='dense2')
    dense_outputs2 = dense_layer2(dense_outputs1)
    output_layer = Dense(n_actions, activation='linear', kernel_initializer='TruncatedNormal', name='output')
    outputs = output_layer(dense_outputs2)
    return Model(inputs=inputs, outputs=outputs)


model = dqn_model()
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
#model.fit(state, pred, batch_size=1)

best_reward = -math.inf
episode_rewards = []
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
        new_state_q_values = model.predict(new_state)
        max_new_state_q_value = np.max(new_state_q_values)
        state_q_values[0, action] = reward + (1 - done) * gamma * max_new_state_q_value
        model.fit(x=state, y=state_q_values, batch_size=1, verbose=0)
        state = new_state
        if done:
            break
    if episode_reward > best_reward:
        best_reward = episode_reward
    episode_rewards.append(episode_reward)
    epsilon = get_epsilon(episode)
    print('Episode {} Reward={} Epsilon={:.4f}'.format(episode, episode_reward, epsilon))
print('Best reward is {}'.format(best_reward))

plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.show()

