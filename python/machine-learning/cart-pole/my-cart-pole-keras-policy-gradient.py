import numpy as np
import gym
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam

learning_rate=0.001

env = gym.make("CartPole-v0")
env = env.unwrapped

n_actions = env.action_space.n

env.reset()

print('num actions={}'.format(n_actions))

def dense_model():
    state_input = Input(shape=[4], name='state_input')
    dense_layer1 = Dense(10, activation='relu', kernel_initializer='he_uniform', name='dense1')
    dense_output1 = dense_layer1(state_input)
    dense_layer2 = Dense(6, activation='relu', kernel_initializer='he_uniform', name='dense2')
    dense_output2 = dense_layer2(dense_output1)
    dense_layer3 = Dense(n_actions, activation='softmax', kernel_initializer='he_uniform', name='dense3')
    dense_output3 = dense_layer3(dense_output2)

    action_input = Input(shape=[n_actions], name='action_input')

    #discount_episode_reward_input = Input(shape=[2], name='reward_input')
    #loss_output =
    return Model(inputs=state_input, outputs=dense_output3)

model = dense_model()
print(model.summary())
model.compile(loss='mse', optimizer=Adam(lr=learning_rate))

observation = env.reset()

model.predict(observation.reshape(-1, 4))