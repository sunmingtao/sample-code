import numpy as np
import gym
from PIL import Image
from keras.models import Model, model_from_config
from keras.layers import Flatten, Conv2D, Input, Dense, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback, CallbackList
import keras.backend as K
import os
import sys
import tensorflow as tf
import timeit
import warnings
from keras.utils.generic_utils import Progbar

sys.path.append('machine-learning/pacman')
from memory import SequentialMemory


VERSION = '001'
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4 # Use 4 frames as input to capture the movement
eps = .1
root_dir = './machine-learning/pacman/'
batch_size = 32
n_warmup_steps = 50000
train_interval = 4
gamma=.99
target_model_update=10000
delta_clip=1.


def preprocess_observation(observation):
    assert observation.ndim == 3
    img = Image.fromarray(observation)
    img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
    processed_observation = np.array(img)
    assert processed_observation.shape == INPUT_SHAPE
    assert processed_observation.dtype == 'uint8'
    return processed_observation

def process_state_batch(batch):
    batch = np.array(batch)
    return batch.astype('float32') / 255.

def process_reward(reward):
    return np.clip(reward, -1., 1.)

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))


env_name = 'MsPacman-v0'
env = gym.make(env_name)
np.random.seed(231) # Everytime generates same random number, for debugging purpose
env.seed(123) # Everytime generates same random number, for debugging purpose
n_actions = env.action_space.n
recent_observation = None
recent_action = None

# Define model

input_shape = (WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1])
inputs = Input(shape=input_shape)
conv_layer1 = Conv2D(32, kernel_size=(8,8), strides=4, activation='relu', data_format='channels_first', name='conv1')
conv_outputs1 = conv_layer1(inputs)
conv_layer2 = Conv2D(64, kernel_size=(4,4), strides=2, activation='relu', data_format='channels_first', name='conv2')
conv_outputs2 = conv_layer2(conv_outputs1)
conv_layer3 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', data_format='channels_first', name='conv3')
conv_outputs3 = conv_layer3(conv_outputs2)
flatten_layer = Flatten(name='flattern')
flatten_outputs = flatten_layer(conv_outputs3)
dense_layer1 = Dense(512, activation='relu', name='dense1')
dense_outputs1 = dense_layer1(flatten_outputs)
dense_layer2 = Dense(n_actions, activation='linear', name='dense2')
dense_outputs2 = dense_layer2(dense_outputs1)
model = Model(inputs=inputs,outputs=dense_outputs2)
print(model.summary())


def clone_model(model):
    # Requires Keras 1.0.7 since get_config has breaking changes.
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config)
    clone.set_weights(model.get_weights())
    return clone

def huber_loss(y_true, y_pred, clip_value):
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if np.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if hasattr(tf, 'select'):
        return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
    else:
        return tf.where(condition, squared_loss, linear_loss)  # condition, true, false


def clipped_masked_error(args):
    y_true, y_pred, mask = args
    loss = huber_loss(y_true, y_pred, delta_clip)
    loss *= mask  # apply element-wise mask
    return K.sum(loss, axis=-1)


def compile(optimizer, metrics=[]):
    metrics += [mean_q]  # register default metrics
    # We never train the target model, hence we can set the optimizer and loss arbitrarily.
    target_model = clone_model(model)
    target_model.compile(optimizer='sgd', loss='mse')
    model.compile(optimizer='sgd', loss='mse')


    # Create trainable model. The problem is that we need to mask the output since we only
    # ever want to update the Q values for a certain action. The way we achieve this is by
    # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
    # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
    y_pred = model.output
    y_true = Input(name='y_true', shape=(n_actions,))
    mask = Input(name='mask', shape=(n_actions,))
    loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_true, y_pred, mask])
    ins = [model.input]
    trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
    assert len(trainable_model.output_names) == 2
    assert trainable_model.output_names[1] == 'dense2'
    combined_metrics = {trainable_model.output_names[1]: metrics}
    losses = [
        lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
        lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
    ]
    trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
    return trainable_model, target_model


def update_target_model():
    target_model.set_weights(model.get_weights())


class MyCallback(Callback):

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass


class MyCallbackList(CallbackList):

    def on_episode_begin(self, episode, logs={}):
        """ Called at beginning of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_begin` callback.
            # If not, fall back to `on_epoch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_begin', None)):
                callback.on_episode_begin(episode, logs=logs)
            else:
                callback.on_epoch_begin(episode, logs=logs)

    def on_episode_end(self, episode, logs={}):
        """ Called at end of each episode for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_episode_end` callback.
            # If not, fall back to `on_epoch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_episode_end', None)):
                callback.on_episode_end(episode, logs=logs)
            else:
                callback.on_epoch_end(episode, logs=logs)

    def on_step_begin(self, step, logs={}):
        """ Called at beginning of each step for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_begin` callback.
            # If not, fall back to `on_batch_begin` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_begin', None)):
                callback.on_step_begin(step, logs=logs)
            else:
                callback.on_batch_begin(step, logs=logs)

    def on_step_end(self, step, logs={}):
        """ Called at end of each step for each callback in callbackList"""
        for callback in self.callbacks:
            # Check if callback supports the more appropriate `on_step_end` callback.
            # If not, fall back to `on_batch_end` to be compatible with built-in Keras callbacks.
            if callable(getattr(callback, 'on_step_end', None)):
                callback.on_step_end(step, logs=logs)
            else:
                callback.on_batch_end(step, logs=logs)

    def on_action_begin(self, action, logs={}):
        """ Called at beginning of each action for each callback in callbackList"""
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_begin', None)):
                callback.on_action_begin(action, logs=logs)

    def on_action_end(self, action, logs={}):
        """ Called at end of each action for each callback in callbackList"""
        for callback in self.callbacks:
            if callable(getattr(callback, 'on_action_end', None)):
                callback.on_action_end(action, logs=logs)


class TrainIntervalLogger(Callback):
    def __init__(self, interval=10000):
        self.interval = interval
        self.step = 0
        self.reset()

    def reset(self):
        """ Reset statistics """
        self.interval_start = timeit.default_timer()
        self.progbar = Progbar(target=self.interval)
        self.metrics = []
        self.infos = []
        self.info_names = None
        self.episode_rewards = []

    def on_train_begin(self, logs):
        """ Initialize training statistics at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = metrics_names()
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training duration at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_step_begin(self, step, logs):
        """ Print metrics if interval is over """
        if self.step % self.interval == 0:
            if len(self.episode_rewards) > 0:
                metrics = np.array(self.metrics)
                assert metrics.shape == (self.interval, len(self.metrics_names))
                formatted_metrics = ''
                if not np.isnan(metrics).all():  # not all values are means
                    means = np.nanmean(self.metrics, axis=0)
                    assert means.shape == (len(self.metrics_names),)
                    for name, mean in zip(self.metrics_names, means):
                        formatted_metrics += ' - {}: {:.3f}'.format(name, mean)

                formatted_infos = ''
                print('{} episodes - episode_reward: {:.3f} [{:.3f}, {:.3f}]{}{}'.format(len(self.episode_rewards), np.mean(self.episode_rewards), np.min(self.episode_rewards), np.max(self.episode_rewards), formatted_metrics, formatted_infos))
                print('')
            self.reset()
            print('Interval {} ({} steps performed)'.format(self.step // self.interval + 1, self.step))

    def on_step_end(self, step, logs):
        """ Update progression bar at the end of each step """
        values = [('reward', logs['reward'])]
        self.progbar.update((self.step % self.interval) + 1, values=values)
        self.step += 1
        self.metrics.append(logs['metrics'])

    def on_episode_end(self, episode, logs):
        """ Update reward value at the end of each episode """
        self.episode_rewards.append(logs['episode_reward'])



class MyCheckPoint(MyCallback):
    def __init__(self, filepath, interval, verbose=0):
        super(MyCheckPoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        self.model.save_weights(filepath, overwrite=True)


class TrainEpisodeLogger(MyCallback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0

    def on_train_begin(self, logs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.metrics_names = metrics_names()
        print('Training for {} steps ...'.format(self.params['nb_steps']))

    def on_train_end(self, logs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_begin(self, episode, logs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[episode] = timeit.default_timer()
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []
        self.metrics[episode] = []

    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)
        nb_step_digits = str(int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps}, steps per second: {sps:.0f}, episode reward: {episode_reward:.3f}, mean reward: {reward_mean:.3f} [{reward_min:.3f}, {reward_max:.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}], mean observation: {obs_mean:.3f} [{obs_min:.3f}, {obs_max:.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': metrics_text,
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, step, logs):
        """ Update statistics of episode after each step """
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.actions[episode].append(logs['action'])
        self.metrics[episode].append(logs['metrics'])
        self.step += 1


def select_action(q_values):
    assert q_values.ndim == 1
    if np.random.uniform() < eps:
        action = np.random.randint(0, n_actions)
    else:
        action = np.argmax(q_values)
    return action


def compute_batch_q_values(state_batch):
    batch = process_state_batch(state_batch)
    q_values = model.predict_on_batch(batch)
    assert q_values.shape == (len(state_batch), n_actions)
    return q_values


def compute_q_values(state):
    q_values = compute_batch_q_values([state]).flatten()
    assert q_values.shape == (n_actions,)
    return q_values


def forward(observation):
    global recent_observation, recent_action
    # Select an action.
    state = memory.get_recent_state(observation)
    q_values = compute_q_values(state)
    action = select_action(q_values=q_values)
    # Book-keeping.
    recent_observation = observation
    recent_action = action
    return action


def metrics_names():
    # Throw away individual losses and replace output name since this is hidden from the user.
    assert len(trainable_model.output_names) == 2
    dummy_output_name = trainable_model.output_names[1]
    model_metrics = [name for idx, name in enumerate(trainable_model.metrics_names) if idx not in (1, 2)]
    model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]
    return model_metrics


def backward(step, reward, terminal):
    # Store most recent experience in memory.
    memory.append(recent_observation, recent_action, reward, terminal)
    metrics = [np.nan for _ in metrics_names()]
    # Train the network on a single stochastic batch.
    if step > n_warmup_steps and step % train_interval == 0:
        experiences = memory.sample(batch_size)
        assert len(experiences) == batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = process_state_batch(state0_batch)
        state1_batch = process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        # Compute the q_values given state1, and extract the maximum for each sample in the batch.
        # We perform this prediction on the target_model instead of the model for reasons
        # outlined in Mnih (2015). In short: it makes the algorithm more stable.
        target_q_values = target_model.predict_on_batch(state1_batch)
        assert target_q_values.shape == (batch_size, n_actions)
        q_batch = np.max(target_q_values, axis=1).flatten()
        assert q_batch.shape == (batch_size,)

        targets = np.zeros((batch_size, n_actions))
        dummy_targets = np.zeros((batch_size,))
        masks = np.zeros((batch_size, n_actions))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        ins = [state0_batch]
        metrics = trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]
        assert len(metrics) == 3
    if step % target_model_update == 0:
        update_target_model()
    return metrics


def fit(callbacks, total_steps, verbose=1):
    callback_list = MyCallbackList(callbacks)
    callback_list.set_model(model)
    params = {
        'nb_steps': total_steps,
    }
    callback_list.set_params(params)
    callback_list.on_train_begin()
    episode = 0
    step = 0
    observation = None
    episode_reward = None
    episode_step = None
    while step < total_steps:
        if observation is None:  # start of a new episode
            callback_list.on_episode_begin(episode)
            episode_reward = 0
            episode_step = 0
            observation = preprocess_observation(env.reset())
        assert episode_reward is not None
        assert episode_step is not None
        assert observation is not None
        callback_list.on_step_begin(episode_step)
        step += 1
        action = forward(observation)
        callback_list.on_action_begin(action)
        observation, reward, done, info = env.step(action)
        observation = preprocess_observation(observation)
        reward = process_reward(reward)
        callback_list.on_action_end(action)
        metrics = backward(step, reward, terminal=done)
        episode_reward += reward
        step_logs = {
            'action': action,
            'observation': observation,
            'reward': reward,
            'metrics': metrics,
            'episode': episode,
        }
        callback_list.on_step_end(episode_step, step_logs)
        episode_step += 1
        if done:
            # We are in a terminal state but the agent hasn't yet seen it. We therefore
            # perform one more forward-backward call and simply ignore the action before
            # resetting the environment. We need to pass in `terminal=False` here since
            # the *next* state, that is the state of the newly reset environment, is
            # always non-terminal by convention.
            forward(observation)
            backward(step, 0., terminal=False)
            episode_logs = {
                'episode_reward': episode_reward,
                'nb_episode_steps': episode_step,
                'nb_steps': step,
            }
            callback_list.on_episode_end(episode, episode_logs)
            episode += 1
            observation = None
            episode_step = None
            episode_reward = None


memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
checkpoint_weights_filename = os.path.join(root_dir, 'my_pacman_weights_weights_{step}.h5f')
callbacks = [MyCheckPoint(checkpoint_weights_filename, interval=100000, verbose=1)]
callbacks += [TrainEpisodeLogger()]
callbacks += [TrainIntervalLogger(interval=10000)]
trainable_model, target_model = compile(Adam(lr=.00025), metrics=['mae'])
fit(callbacks=callbacks, total_steps=10000000, verbose=1)






