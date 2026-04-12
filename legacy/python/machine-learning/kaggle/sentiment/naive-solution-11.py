'''Change log
Based on version 3
Minor refactor. Expect to reproduce the result of version 3
'''
import numpy as np
import tensorflow as tf
import pandas as pd
import re
import os
import sys
import collections
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Conv1D, Input, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import models
from keras.models import Model



BATCH_SIZE = 32

vocabulary_size = 10000

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

def training_data_generator(df, batch_size = BATCH_SIZE):
    X, y = [], []
    while True:
        shuffled_df = df.sample(frac=1)
        for index, data in shuffled_df.iterrows():
            phrase = data['Phrase']
            label = data['Sentiment']
            word_list = preprocess_phrase(phrase)
            num_list = word_to_num(word_list)
            X.append(num_list)
            y.append(label)
            if len(X) >= batch_size:
                yield np.array(sequence.pad_sequences(X, maxlen=longest_phrase_size)), to_categorical(np.array(y), 5)
                X, y = [], []

train_set, validation_set = train_test_split(train_df, test_size=0.25, random_state=42)


n_steps = longest_phrase_size
n_neurons = 300
n_outputs = 5
n_epochs = 8
version = '011'

print('Define model...')


inputs = Input(shape=(n_steps,), name="inputs")
embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=n_neurons, name='embedding')
embedding_output = embedding_layer(inputs)
lstm_layer = LSTM(n_neurons, dropout=0.5, recurrent_dropout=0.5, name='lstm')
lstm_output = lstm_layer(embedding_output)
dense_layer = Dense(n_outputs, activation='softmax', name='dense')
outputs = dense_layer(lstm_output)

model = Model(inputs=inputs, outputs=outputs)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
weight_path="{}_weights.best.hdf5".format('sentiment')
checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

train_steps = len(train_set) // BATCH_SIZE
validation_steps = len(validation_set) // BATCH_SIZE
model.fit_generator(training_data_generator(train_set), steps_per_epoch=train_steps, epochs=n_epochs, validation_data=training_data_generator(validation_set), validation_steps=validation_steps, verbose=1, callbacks=[checkpoint])

print('End training')

model.load_weights(weight_path)
model.save('sentiment_model_{}.h5'.format(version))

model = models.load_model('sentiment_model_{}.h5'.format(version), compile=False)

output = []
for index, value in test_df.iterrows():
    phrase_id = value['PhraseId']
    phrase = value['Phrase']
    word_list = preprocess_phrase(phrase)
    num_list = word_to_num(word_list)
    num_arr = np.array(num_list)
    num_arr = np.expand_dims(num_arr, 0)
    pad_num_arr = sequence.pad_sequences(num_arr, maxlen=longest_phrase_size)
    predict = np.argmax(model.predict(pad_num_arr), axis=-1)
    output += [[phrase_id, predict[0]]]

sub_pd = pd.DataFrame(output)
sub_pd.columns = ['PhraseId','Sentiment']
sub_pd.to_csv('sentiment-{}.csv'.format(version), index=False)
print('Finished')