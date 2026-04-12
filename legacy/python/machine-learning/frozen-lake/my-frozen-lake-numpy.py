import numpy as np
import gym
import random

env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n

q_table = np.zeros((state_size, action_size))

n_max_episodes = 20000       # Total episodes
learning_rate = .9       # Learning rate
n_max_steps = 100            # Max steps per episode
gamma = 0.95                # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005             # Exponential decay rate for exploration prob


def get_action(observation):
    # 3. Choose an action a in the current world state (s)
    ran = random.uniform(0, 1)
    if ran > epsilon:
        return np.argmax(q_table[observation, :])
    else:
        return env.action_space.sample()

def get_epsilon(episode):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


for gamma in [.995]:
    for learning_rate in [1, .99, .9, .8, .5, .3,  .1, .05, .01]:
        total_avg = 0
        for i in range(3):
            q_table = np.zeros((state_size, action_size))
            epsilon = 1.0
            episode_rewards = []
            # 2 For life or until learning is stopped
            for episode in range(n_max_episodes):
                # Reset the environment
                observation = env.reset()
                step = 0
                done = False
                total_reward = 0

                for step in range(n_max_steps):

                    action = get_action(observation)

                    # Take the action (a) and observe the outcome state(s') and reward (r)
                    new_observation, reward, done, info = env.step(action)
                    total_reward += reward
                    old_observation_q_value = q_table[observation, action]
                    q_table[observation, action] = old_observation_q_value + learning_rate * (reward + gamma * np.max(q_table[new_observation, :]) - old_observation_q_value)
                    # Our new state is state
                    observation = new_observation

                    # If done (if we're dead) : finish episode
                    if done:
                        break
                if total_reward > 0:
                    epsilon = get_epsilon(episode)
                episode_rewards.append(total_reward)
                #print('Episode {}, Total reward = {}, epsilon = {:.2f}'.format(episode, total_reward, epsilon))

            #print(q_table)
            total_avg += sum(episode_rewards) / len(episode_rewards)
            print('Average reward', sum(episode_rewards) / len(episode_rewards))

        print('Gamma={}, Learning Rate={}, Average Average={:.3f}'.format(gamma, learning_rate, total_avg / 3))