import numpy as np
import tensorflow as tf

np.random.seed(42)

default_reber_grammar = [
    [("B", 1)],           # (state 0) =B=>(state 1)
    [("T", 2), ("P", 3)], # (state 1) =T=>(state 2) or =P=>(state 3)
    [("S", 2), ("X", 4)], # (state 2) =S=>(state 2) or =X=>(state 4)
    [("T", 3), ("V", 5)], # and so on...
    [("X", 3), ("S", 6)],
    [("P", 4), ("V", 6)],
    [("E", None)]]        # (state 6) =E=>(terminal state)

embedded_reber_grammar = [
    [("B", 1)],
    [("T", 2), ("P", 3)],
    [(default_reber_grammar, 4)],
    [(default_reber_grammar, 5)],
    [("T", 6)],
    [("P", 6)],
    [("E", None)]]


def generate_string(grammar):
    state = 0
    output = []
    while state is not None:
        index = np.random.randint(len(grammar[state]))
        production, state = grammar[state][index]
        if isinstance(production, list):
            production = generate_string(grammar=production)
        output.append(production)
    return "".join(output)


def generate_corrupted_string(grammar, chars="BEPSTVX"):
    good_string = generate_string(grammar)
    index = np.random.randint(len(good_string))
    good_char = good_string[index]
    bad_char = np.random.choice(sorted(set(chars) - set(good_char)))
    return good_string[:index] + bad_char + good_string[index + 1:]


for _ in range(25):
    print(generate_string(default_reber_grammar))

for _ in range(25):
    print(generate_string(embedded_reber_grammar))

for _ in range(25):
    print(generate_corrupted_string(embedded_reber_grammar))


def string_to_one_hot_vectors(string, n_steps, chars="BEPSTVX"):
    char_to_index = {char: index for index, char in enumerate(chars)}
    output = np.zeros((n_steps, len(chars)), dtype=np.int32)
    for index, char in enumerate(string):
        output[index, char_to_index[char]] = 1.
    return output


def generate_dataset(size):
    good_strings = [generate_string(embedded_reber_grammar) for _ in range(size // 2)]
    bad_strings = [generate_corrupted_string(embedded_reber_grammar) for _ in range(size - size // 2)]
    all_strings = good_strings + bad_strings
    n_steps = max([len(string) for string in all_strings])
    X = np.array([string_to_one_hot_vectors(string, n_steps) for string in all_strings])
    seq_length = np.array([len(string) for string in all_strings])
    y = np.array([[1] for _ in range(len(good_strings))] + [[0] for _ in range(len(bad_strings))])
    rnd_idx = np.random.permutation(size)
    #  (size, n_steps, 7), (size, ),        (size, 1)
    return X[rnd_idx], seq_length[rnd_idx], y[rnd_idx]


X_train, l_train, y_train = generate_dataset(10000)

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

possible_chars = "BEPSTVX"
n_inputs = len(possible_chars)
n_neurons = 30
n_outputs = 1

learning_rate = 0.02
momentum = 0.95

X = tf.placeholder(tf.float32, [None, None, n_inputs], name="X") #[batch_size, steps, n_inputs]
seq_length = tf.placeholder(tf.int32, [None], name="seq_length") #[steps]
y = tf.placeholder(tf.float32, [None, 1], name="y") #[batch_size, 1]

gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32, sequence_length=seq_length)
#What's the shape of outputs? [batch_size, steps, n_neurons] - Corresponds to each letter
#What's the shape of states? [batch size, n_neurons]

logits = tf.layers.dense(states[0], n_outputs, name="logits")
y_pred = tf.cast(tf.greater(logits, 0.5), tf.float32, name="y_pred")
y_proba = tf.nn.sigmoid(logits, name="y_proba")
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)
correct = tf.equal(y_pred, y, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

X_val, l_val, y_val = generate_dataset(5000)

n_epochs = 50
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        X_batches = np.array_split(X_train, len(X_train) // batch_size)
        l_batches = np.array_split(l_train, len(l_train) // batch_size)
        y_batches = np.array_split(y_train, len(y_train) // batch_size)
        for X_batch, l_batch, y_batch in zip(X_batches, l_batches, y_batches):
            loss_val, _ = sess.run( [loss, training_op], feed_dict={X: X_batch, seq_length: l_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, seq_length: l_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_val, seq_length: l_val, y: y_val})
        print("{:4d}  Train loss: {:.4f}, accuracy: {:.2f}%  Validation accuracy: {:.2f}%".format(epoch, loss_val, 100 * acc_train, 100 * acc_val))
        saver.save(sess, "./my_reber_classifier")


test_strings = ["BPBTSSSSSSSXXTTVPXVPXTTTTTVVETE", "BPBTSSSSSSSXXTTVPXVPXTTTTTVVEPE", "BPBTSSSSSSSXXTTVPXVPXTTTTTVVEPE", "BPBTSSSSSSSXXTTVPXVPXTTTTTVVEPE"]
l_test = np.array([len(s) for s in test_strings])
max_length = l_test.max()
X_test = [string_to_one_hot_vectors(s, n_steps=max_length) for s in test_strings]

with tf.Session() as sess:
    saver.restore(sess, "./my_reber_classifier")
    y_proba_val, y_pred_val = sess.run([y_proba, y_pred], feed_dict={X: X_test, seq_length: l_test})

print("Estimated probability that these are Reber strings:")
for index, string in enumerate(test_strings):
    print("{}: {:.2f}%, {}".format(string, 100 * y_proba_val[index][0], y_pred_val[index][0]))


#Test code
with tf.Session() as sess:
    init.run()
    states_val = sess.run([states], feed_dict={X: X_test, seq_length: l_test})

np.array(states_val).shape