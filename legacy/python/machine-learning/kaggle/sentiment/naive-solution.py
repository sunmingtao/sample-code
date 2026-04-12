import numpy as np
import tensorflow as tf
import pandas as pd
import re
import os
import sys
import collections
from sklearn.model_selection import train_test_split



BATCH_SIZE = 16

vocabulary_size = 200

root_dir = os.path.join(os.getcwd(), 'machine-learning/kaggle/sentiment')
train_csv = os.path.join(root_dir, 'train.tsv')
test_csv = os.path.join(root_dir, 'test.tsv')
train_df = pd.read_csv(train_csv, sep='\t')
test_df = pd.read_csv(test_csv, sep='\t')

#Group by sentence Id, get the min phrase Id
min_phrase_id_series = train_df.groupby('SentenceId')['PhraseId'].idxmin() # Index of min phrase of each sentence
min_phrase_id_df = train_df.iloc[min_phrase_id_series.values]
all_phrase_list = min_phrase_id_df['Phrase'].values

#Size of longest phrase
longest_phrase_size = max(len(phrase.split()) for phrase in all_phrase_list)

#Gather all the words and convert to lower case
all_words_str = ' '.join(all_phrase_list)
all_words_str = all_words_str.lower()
all_words = sorted(set(all_words_str.split()))
all_words = [word.strip() for word in all_words]

#Retain only words that start with [a-z]
all_words = [word for word in all_words if word[0].isalpha()]

def preprocess_phrase(phrase):
    phrase_list = phrase.lower().split()
    phrase_list = [phrase.strip() for phrase in phrase_list] #Trim leading and trailing spaces
    phrase_list = [phrase for phrase in phrase_list if phrase[0].isalpha()] #Retain only alpha words
    return phrase_list


# Create vocabulary dictionary
all_phrase_list = min_phrase_id_df['Phrase'].values
all_words_str = ' '.join(all_phrase_list)
all_words_str = all_words_str.lower()
all_words_list = all_words_str.split()
all_words_list = [word.strip() for word in all_words_list]
all_words_list = [word for word in all_words_list if word[0].isalpha()] #Retain only alpha words
counter_list = collections.Counter(all_words_list).most_common(vocabulary_size-1) # -1 because one slot is reserved for [UNK] - Unknown
'''[('the', 7233),
 ('a', 5234),
 ('and', 4426),
 ('of', 4340),
 ('to', 2996),
 ('is', 2538),
 ('it', 2405),
 ('that', 1937),
 ('in', 1870),
 ('as', 1281),
 ('but', 1168)....] '''

word_index_dict = {value[0]: index+1 for index, value in enumerate(counter_list)}
'''{'the': 1,
 'a': 2,
 'and': 3,
 'of': 4,
 'to': 5,
 'is': 6,
 'it': 7,
 'that': 8,
 'in': 9,
 'as': 10...'''

def get_index_by_word(word):
    if word in word_index_dict:
        return word_index_dict[word]
    else:
        return 0

def word_to_num(word_list):
    return [get_index_by_word(word) for word in word_list]

def num_to_one_hot_vector(num_list):
    output = np.zeros((longest_phrase_size, vocabulary_size), dtype=np.int32)
    for index, value in enumerate(num_list):
        output[index, value] = 1
    return output

def label_to_one_hot_vector(label):
    output = np.zeros(5, dtype=np.int32)
    output[label] = 1
    return output

def phrase_to_one_hot_vector(phrase):
    word_list = preprocess_phrase(phrase)
    num_list = word_to_num(word_list)
    return num_to_one_hot_vector(num_list)

#TODO Change it to generator because generating everything in one go takes too long
def generate_training_data(df):
    X , y, seq_length = [], [], []
    for index, data in df.iterrows():
        phrase = data['Phrase']
        label = data['Sentiment']
        X.append(phrase_to_one_hot_vector(phrase))
        y.append(label)
        word_list = preprocess_phrase(phrase)
        seq_length.append(len(word_list))
    return X, y, seq_length


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

n_steps = longest_phrase_size
n_neurons = 64
n_outputs = 5

learning_rate = 0.02
momentum = 0.95

X = tf.placeholder(tf.float32, [None, n_steps, vocabulary_size], name="X") #[batch_size, n_steps, vocabulary_size]
seq_length = tf.placeholder(tf.int32, [None], name="seq_length") #[seq_length] - not fixed value
y = tf.placeholder(tf.int32, name="y") #[batch_size]

gru_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(gru_cell, X, dtype=tf.float32, sequence_length=seq_length)

logits = tf.layers.dense(states, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

train_set, validation_set = train_test_split(train_df, test_size=0.3, random_state=42)
X_data, y_data, seq_length_data = generate_training_data(train_set)
X_val_data, y_val_data, seq_length_val_data = generate_training_data(validation_set)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        X_batches = np.array_split(X_data, len(X_data) // batch_size)
        y_batches = np.array_split(y_data, len(y_data) // batch_size)
        seq_length_batches = np.array_split(seq_length_data, len(seq_length_data) // batch_size)
        for X_batch, y_batch, seq_length_batch, index in zip(X_batches, y_batches, seq_length_batches, range(len(X_batches))):
            loss_val, _ = sess.run([loss, training_op], feed_dict={X: X_batch, seq_length: seq_length_batch, y: y_batch})
            if index % 1000 == 0:
                print('Processed {}'.format(index))
        acc_train = accuracy.eval(feed_dict={X: X_data, seq_length: seq_length_data, y: y_data})
        acc_val = accuracy.eval(feed_dict={X: X_val_data, seq_length: seq_length_val_data, y: y_val_data})
        print("{:4d}  Train loss: {:.4f}, accuracy: {:.2f}%  Validation accuracy: {:.2f}%".format(epoch, loss_val, 100 * acc_train, 100 * acc_val))
        saver.save(sess, "./my_sentiment_classifier")


X_test = [phrase_to_one_hot_vector(phrase) for phrase in test_df['Phrase'].values]
seq_length_test = [len(preprocess_phrase(phrase)) for phrase in test_df['Phrase'].values]
phrase_ids = test_df['PhraseId'].values

output = []
with tf.Session() as sess:
    saver.restore(sess, "./my_sentiment_classifier2")
    logits_val = sess.run([logits], feed_dict={X: X_test, seq_length: seq_length_test})
    print('logits_val done')
    for phrase_id, pred in zip(phrase_ids, logits_val[0]):
        p = np.argmax(pred)
        output += [[phrase_id, p]]

sub_pd = pd.DataFrame(output)
sub_pd.columns = ['PhraseId','Sentiment']
sub_pd.to_csv('sentiment-001.csv', index=False)
