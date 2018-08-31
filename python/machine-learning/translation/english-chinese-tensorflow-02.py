'''
Based on version 1.
Pad 0s to the end instead of front
'''
import os
import re
import collections
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time

# Constants
BATCH_SIZE = 16

# Read data
root_dir = os.path.join(os.getcwd(), 'machine-learning/translation')
data_path = os.path.join(root_dir, 'cmn.txt')

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

input_texts = []
output_texts = []
try:
    for line in lines:
        input_text, output_text = line.split('\t')
        input_texts.append(input_text)
        output_texts.append('{'+output_text+'}') # { denotes the start of sentence, } denotes the end of sentence
except ValueError:
    print(line)

# Preprocess
def remove_blank(word_list):
    return [word for word in word_list if word.strip()]

def split_input_texts(text):
    input_text_list = re.split("([^a-z0-9'-])", text.lower())
    return remove_blank(input_text_list)

# Count unique number of English words
all_input_texts = ' '.join(input_texts)
all_input_list = split_input_texts(all_input_texts)
input_vocabulary_size = len(set(all_input_list)) + 1  # 6285

# Make word-index lookup for english words
input_counter_list = collections.Counter(all_input_list).most_common(input_vocabulary_size-1)
input_word_index_dict = {value[0]: index+1 for index, value in enumerate(input_counter_list)}
input_index_word_dict = {index+1: value[0] for index, value in enumerate(input_counter_list)}

# Count unique number of Chinese characters
all_output_texts = ''.join(output_texts)
output_vocabulary_size = len(set(all_output_texts)) + 1  # 3418

# Make word-index lookup for chinese words
output_counter_list = collections.Counter(all_output_texts).most_common(output_vocabulary_size-1)
output_word_index_dict = {value[0]: index+1 for index, value in enumerate(output_counter_list)}
output_index_word_dict = {index+1: value[0] for index, value in enumerate(output_counter_list)}
output_index_word_dict[0] = ''

# Max size for input and output text
input_max_size = max(len(split_input_texts(text)) for text in input_texts)  # 34
output_max_size = max(len(text) for text in output_texts)  # 46


def input_text_to_num(input_text):
    input_text_list = split_input_texts(input_text)
    num_list = [input_word_index_dict[word] for word in input_text_list]
    return num_list

def output_text_to_num(output_text):
    num_list = [output_word_index_dict[character] for character in output_text]
    return num_list

def output_num_to_text(num_list):
    text = [output_index_word_dict[num] for num in num_list]
    return ''.join(text)

def pad_num_list(num_list, size):
    return num_list + [0] * (size - len(num_list))

def input_text_to_x(input_text):
    num_list = input_text_to_num(input_text)
    return pad_num_list(num_list, input_max_size)

def output_text_to_y(output_text):
    num_list = output_text_to_num(output_text)
    return pad_num_list(num_list, output_max_size)


# Convert input/output texts to machine learning input X/y
X = np.array([input_text_to_x(text) for text in input_texts])
y = np.array([output_text_to_y(text) for text in output_texts])

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def batch_generator(x, y, batch_size=BATCH_SIZE):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size

# Build model
embed_size = 300
n_neurons = 150

tf.reset_default_graph()

inputs = tf.placeholder(tf.int64, (None, input_max_size), 'inputs')
outputs = tf.placeholder(tf.int64, (None, None), 'output')
targets = tf.placeholder(tf.int64, (None, None), 'targets')

input_embedding = tf.Variable(tf.random_uniform((input_vocabulary_size, embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform((output_vocabulary_size, embed_size), -1.0, 1.0), name='dec_embedding')

input_embed = tf.nn.embedding_lookup(input_embedding, inputs) # (batch_size, input_max_size, embed_size)
output_embed = tf.nn.embedding_lookup(output_embedding, outputs) # (batch_size, output_max_size, embed_size)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_encode = tf.contrib.rnn.BasicLSTMCell(n_neurons)
    encode_outputs, encode_state = tf.nn.dynamic_rnn(lstm_encode, inputs=input_embed, dtype=tf.float32)
    # encode_outputs.shape = (batch_size, input_max_size, n_neurons)
    # encode_state.shape = (2, batch_size, n_neurons)

with tf.variable_scope("decoding") as decoding_scope:
    lstm_decode = tf.contrib.rnn.BasicLSTMCell(n_neurons)
    decode_outputs, decode_state = tf.nn.dynamic_rnn(lstm_decode, initial_state=encode_state, inputs=output_embed, dtype=tf.float32)
    # decode_outputs.shape = (batch_size, output_max_size, n_neurons)
    # decode_state.shape = (2, batch_size, n_neurons)

logits = tf.contrib.layers.fully_connected(inputs=decode_outputs, num_outputs=output_vocabulary_size, activation_fn=None) # (batch_size, output_max_length, output_vocabulary_size)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=-1), targets), tf.float32))

with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([BATCH_SIZE, output_max_size-1]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# Train
saver_path = "./english-chinese-translate-02"
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epochs = 3
saver = tf.train.Saver()
for epoch_i in range(epochs):
    start_time = time.time()
    total_accuracy = 0
    for batch_i, (source_batch, target_batch) in enumerate(batch_generator(X_train, y_train)):
        _, batch_loss, batch_logits, accuracy_val = sess.run([optimizer, loss, logits, accuracy], feed_dict = {inputs: source_batch, outputs: target_batch[:, :-1], targets: target_batch[:, 1:]})
        total_accuracy += accuracy_val
        print('Epoch {}, Batch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_i, batch_loss, total_accuracy / (batch_i + 1), time.time() - start_time))
    accuracy_val = sess.run(accuracy, feed_dict = {inputs: X_test, outputs: y_test[:, :-1], targets: y_test[:, 1:]})
    print('Epoch {}, validation accuracy: {:>6.4f}'.format(epoch_i, accuracy_val))
    saver.save(sess, saver_path)
    print('Training finished')


saver.restore(sess, saver_path)
test_input_texts = input_texts[10003:10013]
X_test_batch = [input_text_to_x(input) for input in test_input_texts]

dec_input = np.zeros((len(X_test_batch), 1)) + output_word_index_dict['{']

# Inference
for i in range(output_max_size):
    logits_eval = sess.run(logits, feed_dict={inputs: X_test_batch, outputs: dec_input})
    prediction = logits_eval[:, -1].argmax(axis=-1)
    dec_input = np.hstack([dec_input, prediction[:, None]])

dec_input = np.array(dec_input, dtype=np.int32)
for eng, chi in zip(test_input_texts, [output_num_to_text(num_list) for num_list in dec_input]):
    print(eng, chi)

# Test code
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

x = tf.placeholder(tf.int64, (None, 2), 'x')
y = tf.placeholder(tf.int64, (None), 'y')
x_argmax = tf.argmax(x, axis=-1)
x_argmax_cast = tf.cast(tf.equal(x_argmax, y), tf.float32)
x_reduce_mean = tf.reduce_mean(x_argmax_cast)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

x_feed = np.array([[1,2],[3,4],[7,6]])
y_feed = np.array([1,1,1])
feed_dict = {x:x_feed, y:y_feed}
x_argmax_val, x_argmax_cast_val, x_reduce_mean_val = sess.run([x_argmax, x_argmax_cast, x_reduce_mean], feed_dict=feed_dict)


