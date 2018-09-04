import gym
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os


env = gym.make("CartPole-v0")

max_steps = 10000

'''
for i in range(20):
    env.reset()
    total_reward = 0
    for step in range(max_steps):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            print('Step {}, Reward {}'.format(step, total_reward))
            break
'''
n_outputs = env.action_space.n


def q_network(X_state, name):
    inputs = X_state
    with tf.variable_scope(name) as scope:
        dense_outputs = tf.layers.dense(inputs, 100, tf.nn.relu, kernel_initializer=tf.variance_scaling_initializer())
        outputs = tf.layers.dense(dense_outputs, n_outputs, kernel_initializer=tf.variance_scaling_initializer())
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return outputs, trainable_vars_by_name


learning_rate = 0.001
momentum = 0.95


tf.reset_default_graph()
X_state = tf.placeholder(tf.float32, shape=[None, 4])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=None) # shape = (batch_size, )
    target_q_value = tf.placeholder(tf.float32, shape=[None, 1]) # shape = (batch_size, 1)
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs), axis=1, keepdims=True) # shape = (batch_size, 1), without keepdims = True, shape = (batch_size, )
    error = tf.abs(target_q_value - q_value) # shape = (batch_size, 1)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=np.object)
        self.index = 0
        self.length = 0

    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size)  # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]


replay_memory_size = 500000
replay_memory = ReplayMemory(replay_memory_size)

def sample_memories(batch_size):
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for memory in replay_memory.sample(batch_size):
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000
game_length = 0
total_max_q = 0

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

checkpoint_path = "./my_cart-pole.ckpt"
n_steps = 10000  # total number of training steps
iteration = 0  # game iterations
loss_val = np.infty
mean_max_q = 0.0
skip_start = 90
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
batch_size = 50
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.998
done = True # env needs to be reset

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        if step % 1000 == 0:
            print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   ".format(iteration, step, n_steps, step * 100 / n_steps, loss_val, mean_max_q), end="")
        if done:  # game over, start again
            obs = env.reset()
            state = obs
        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        obs, reward, done, info = env.step(action)
        next_state = obs

        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        total_max_q += q_values.max()
        game_length += 1

        if done:
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if iteration < training_start or iteration % training_interval != 0:
            continue  # only train after warmup period and at regular intervals

        X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
        next_q_values = target_q_values.eval(feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values
        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={X_state: X_state_val, X_action: X_action_val, target_q_value: y_val})

        # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)


# Play game, random strategy
steps, total_rewards = [] , []
for game in range(10):
    env.reset()
    total_reward = 0
    for step in range(10000):
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if done:
            break
    steps.append(step)
    total_rewards.append(total_reward)
    print('Game {}: Step = {}, Reward = {}'.format(game+1, step, total_reward))
print('Avg steps {}, Avg regards {}'.format(sum(steps)/len(steps), sum(total_rewards)/len(total_rewards)))


# Play game, ML
steps, total_rewards = [] , []
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    for game in range(10):
        obs = env.reset()
        total_reward = 0
        for step in range(10000):
            state = obs
            # Online DQN evaluates what to do
            q_values = online_q_values.eval(feed_dict={X_state: [state]})
            action = np.argmax(q_values)
            # Online DQN plays
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        steps.append(step)
        total_rewards.append(total_reward)
        print('Game {}: Step = {}, Reward = {}'.format(game + 1, step, total_reward))
    print('Avg steps {}, Avg regards {}'.format(sum(steps) / len(steps), sum(total_rewards) / len(total_rewards)))

env.reset()
screen = env.render(mode='rgb_array')

plt.imshow(screen)
plt.show()


for i in range(4):
    print(env.observation_space.high[i],env.observation_space.low[i])
#.transpose((2, 0, 1))



