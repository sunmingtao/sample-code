import numpy as np
import gym
import random

env = gym.make("Taxi-v2")

n_actions = env.action_space.n
n_states = env.observation_space.n

print('num actions={}, num states={}'.format(n_actions, n_states))

q_table = np.zeros((n_states, n_actions))

n_max_episodes = 100000       # Total episodes
n_max_steps = 100            # Max steps per episode

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

n_try = 1
for gamma in [.3, .1]:
    for learning_rate in [.99, .9, .8, .5, .4, .3, .2, .1, .05, .01]:
        total_avg = 0
        for i in range(n_try):
            q_table = np.zeros((n_states, n_actions))
            epsilon = 1.0
            episode_rewards = []
            for episode in range(n_max_episodes):
                # Reset the environment
                observation = env.reset()
                step = 0
                total_reward = 0
                for step in range(n_max_steps):

                    action = get_action(observation)
                    new_observation, reward, done, info = env.step(action)
                    total_reward += reward
                    old_observation_q_value = q_table[observation, action]
                    q_table[observation, action] = old_observation_q_value + learning_rate * (reward + gamma * np.max(q_table[new_observation, :]) - old_observation_q_value)
                    # Our new state is state
                    observation = new_observation

                    # If done (if we're dead) : finish episode
                    if done:
                        break
                epsilon = get_epsilon(episode)
                episode_rewards.append(total_reward)
                #print('Episode {}, Total reward = {}, epsilon = {:.2f}'.format(episode, total_reward, epsilon))

            #print(q_table)
            total_avg += sum(episode_rewards) / len(episode_rewards)
            print('Average reward', sum(episode_rewards) / len(episode_rewards))

        print('Gamma={}, Learning Rate={}, Average Average={:.3f}'.format(gamma, learning_rate, total_avg / n_try))

