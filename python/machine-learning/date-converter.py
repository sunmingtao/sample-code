from faker import Faker
import random
import babel
from babel.dates import format_date
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

fake = Faker()
fake.seed(42)
random.seed(42)

FORMATS = ['short',
           'medium',
           'long',
           'full',
           'd MMM YYY',
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY',
           ]

def create_date():
    """
        Creates some fake dates
        :returns: tuple containing
                  1. human formatted string
                  2. machine formatted string
                  3. date object.
    """
    dt = fake.date_object()

    # wrapping this in a try catch because
    # the locale 'vo' and format 'full' will fail
    try:
        human = format_date(dt, format=random.choice(FORMATS))

        case_change = random.randint(0,3) # 1/2 chance of case change
        if case_change == 1:
            human = human.upper()
        elif case_change == 2:
            human = human.lower()

        machine = dt.isoformat()
    except AttributeError as e:
        return None, None, None

    return human, machine #, dt

data = [create_date() for _ in range(50000)]

x = [x for x, y in data]
y = [y for x, y in data]

u_characters = set(' '.join(x))
'''
{' ',
 ',',
 '.',
 '/',
 '0',
 '1',
 '2',
 '3',
 '4',
 '5',
 '6',
 '7',
 '8',
 '9',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',...'''
char2numX = dict(zip(u_characters, range(len(u_characters))))
'''
{'h': 0,
 '5': 1,
 '2': 2,
 'B': 3,
 'F': 4,
 'I': 5,
 'b': 6,
 'a': 7,
 'o': 8,
 'M': 9,
 'C': 10,
 '9': 11,
 ' ': 12,
 'm': 13,
 'f': 14,
 't': 15...'''

u_characters = set(' '.join(y))
char2numY = dict(zip(u_characters, range(len(u_characters))))

char2numX['<PAD>'] = len(char2numX) #{'<PAD>': 58}
num2charX = dict(zip(char2numX.values(), char2numX.keys()))
'''
{0: 'h',
 1: '5',
 2: '2',
 3: 'B',
 4: 'F',
 5: 'I',
 6: 'b',
 7: 'a',
 8: 'o',
 9: 'M',
 10: 'C',
 11: '9',
 12: ' ',
 13: 'm',
'''

max_len = max([len(date) for date in x])
x = [[char2numX['<PAD>']]*(max_len - len(date)) +[char2numX[x_] for x_ in date] for date in x]
'''
 [58, --<PAD>
  58,
  58,
  58,
  58,
  58,
  58,
  58,
  4,
  3,
  2,
'''
x = np.array(x)


char2numY['<GO>'] = len(char2numY)
num2charY = dict(zip(char2numY.values(), char2numY.keys()))

y = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y]
print(''.join([num2charY[y_] for y_ in y[4]]))
y = np.array(y)

x_seq_length = len(x[0])
y_seq_length = len(y[0])- 1

def batch_data(x, y, batch_size):
    shuffle = np.random.permutation(len(x))
    start = 0
    x = x[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(x):
        yield x[start:start+batch_size], y[start:start+batch_size]
        start += batch_size


epochs = 2
batch_size = 128
nodes = 32
embed_size = 10


tf.reset_default_graph()

inputs = tf.placeholder(tf.int32, (None, x_seq_length), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

input_embedding = tf.Variable(tf.random_uniform((len(char2numX), embed_size), -1.0, 1.0), name='enc_embedding') # (59, embed_size)
output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding') # (13, embed_size)

date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs) # (batch_size, input_max_length, embed_size)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs) # (batch_size, output_max_length, embed_size)

with tf.variable_scope("encoding") as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(nodes)
    enc_outputs, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=date_input_embed, dtype=tf.float32)
    # enc_outputs.shape = (batch_size, input_max_length, nodes)
    # last_state.shape = (2, batch_size, nodes)


with tf.variable_scope("decoding") as decoding_scope:
    # you will need to set initial_state=last_state from the encoder
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, initial_state=last_state, inputs=date_output_embed, dtype=tf.float32)
    # dec_outputs.shape = (batch_size, output_max_length, nodes)
    # _.shape = (2, batch_size, nodes)


logits = tf.contrib.layers.fully_connected(inputs=dec_outputs, num_outputs=len(char2numY), activation_fn=None) # (batch_size, output_max_length, 13)

with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
epochs = 10
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits], feed_dict = {inputs: source_batch, outputs: target_batch[:, :-1], targets: target_batch[:, 1:]})
    accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss, accuracy, time.time() - start_time))

source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))

dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>'] #[batch_size * 1]

for i in range(y_seq_length):
    batch_logits = sess.run(logits, feed_dict={inputs: source_batch, outputs: dec_input})
    prediction = batch_logits[:, -1].argmax(axis=-1)
    print(batch_logits.shape, prediction.shape)
    dec_input = np.hstack([dec_input, prediction[:, None]])

batch_logits.shape

print('Accuracy on test set is: {:>6.3f}'.format(np.mean(dec_input == target_batch)))


num_preds = 10
source_chars = [[num2charX[l] for l in sent if num2charX[l]!="<PAD>"] for sent in source_batch[:num_preds]]
dest_chars = [[num2charY[l] for l in sent] for sent in dec_input[:num_preds, 1:]]

for date_in, date_out in zip(source_chars, dest_chars):
    print(''.join(date_in)+' => '+''.join(date_out))

source_batch, target_batch = next(batch_data(X_train, y_train, batch_size))
logits_val, dec_outputs_val, date_output_embed_val, enc_outputs_val, last_state_val = sess.run([logits, dec_outputs, date_output_embed, enc_outputs, last_state], feed_dict = {inputs: source_batch, outputs: target_batch[:, :-1], targets: target_batch[:, 1:]})
len(date_output_embed_val)
print(date_output_embed_val[0].shape)
np.array(source_batch).shape
target_batch.shape


#Test code
tf.reset_default_graph()
inputs = tf.placeholder(tf.float32, (None, None, 2), 'inputs')
#lstm = tf.contrib.rnn.BasicLSTMCell(2)
gru_cell = tf.contrib.rnn.GRUCell(88)
outputs, state = tf.nn.dynamic_rnn(gru_cell, inputs=inputs, dtype=tf.float32)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

input_val = np.ones((300, 188, 2))
state_val = sess.run(state, feed_dict={inputs: input_val})
state_val.shape
len(state_val)