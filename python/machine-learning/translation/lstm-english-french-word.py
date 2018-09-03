from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu

import os
import re
import collections
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import RegexpTokenizer, word_tokenize
from keras.utils.np_utils import to_categorical
from keras import models

# Constants
BATCH_SIZE = 16
SAMPLE = 20000
VERSION = '001'

# Read data
root_dir = os.path.join(os.getcwd(), 'machine-learning/translation')
data_path = os.path.join(root_dir, 'fra.txt')

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

input_texts = input_texts[:SAMPLE]
output_texts = output_texts[:SAMPLE]

# Preprocess
def remove_blank(word_list):
    return [word for word in word_list if word.strip()]

def is_curly_bracket(string):
    return string == '{' or string == '}'

def remove_blank_curly_bracket(word_list):
    return [word for word in word_list if word.strip() and not is_curly_bracket(word)]

def split_input_texts(text):
    return [word.lower() for word in word_tokenize(text)]

def split_output_texts(text):
    return [word.lower() for word in word_tokenize(text, language='french')]

# Tokenize input and output
input_text_list = [split_input_texts(input_text) for input_text in input_texts]
output_text_list = [split_output_texts(output_text) for output_text in output_texts]

# Count unique number of English words
input_text_flat_list = []
for input_text in input_text_list:
    input_text_flat_list.extend(input_text)

input_vocabulary_size = len(set(input_text_flat_list)) + 1  # 3442
print('input vocabulary size', input_vocabulary_size)

# Make word-index lookup for english words
input_counter_list = collections.Counter(input_text_flat_list).most_common()
input_word_index_dict = {value[0]: index+1 for index, value in enumerate(input_counter_list)}
input_index_word_dict = {index+1: value[0] for index, value in enumerate(input_counter_list)}

# Count unique number of French words
output_text_flat_list = []
for output_text in output_text_list:
    output_text_flat_list.extend(output_text)
output_vocabulary_size = len(set(output_text_flat_list)) + 1  # 7251
print('output vocabulary size', output_vocabulary_size)

# Make word-index lookup for French words
output_counter_list = collections.Counter(output_text_flat_list).most_common(output_vocabulary_size)
output_word_index_dict = {value[0]: index+1 for index, value in enumerate(output_counter_list)}
output_index_word_dict = {index+1: value[0] for index, value in enumerate(output_counter_list)}
output_index_word_dict[0] = ''

# Max size for input and output text
input_max_size = max(len(split_input_texts(text)) for text in input_texts)  # 7
output_max_size = max(len(split_output_texts(text)) for text in output_texts) # 15

print('Input max size', input_max_size)
print('Output max size', output_max_size)

# Convert input/output texts to machine learning input X/y

def input_text_to_num(input_text):
    input_text_list = split_input_texts(input_text)
    num_list = [input_word_index_dict[word] for word in input_text_list]
    return num_list

def output_text_to_num(output_text):
    output_text_list = split_output_texts(output_text)
    num_list = [output_word_index_dict[word] for word in output_text_list]
    return num_list

def output_num_to_text(num_list):
    text = [output_index_word_dict[num] for num in num_list]
    return ' '.join(text)

def pad_num_list(num_list, size):
    return num_list + [0] * (size - len(num_list))

def input_text_to_x(input_text):
    num_list = input_text_to_num(input_text)
    return pad_num_list(num_list, input_max_size)

def output_text_to_y(output_text):
    num_list = output_text_to_num(output_text)
    return pad_num_list(num_list, output_max_size)

X = np.array([input_text_to_x(text) for text in input_texts])
y = np.array([output_text_to_y(text) for text in output_texts])

print("X shape: {}, y shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def batch_generator(x, y, batch_size=BATCH_SIZE):
    while True:
        shuffle = np.random.permutation(len(x))
        start = 0
        x = x[shuffle]
        y = y[shuffle]
        while start + batch_size <= len(x):
            x_batch = x[start:start+batch_size]
            y_batch = y[start:start+batch_size]
            yield [x_batch, y_batch[:,:-1]], to_categorical(y_batch[:, 1:], output_vocabulary_size)
            start += batch_size

#[x_sample, y_sample], y_sample_2 = next(batch_generator(X_train, y_train))

# Define Model
embed_size = 300

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embed_layer = Embedding(input_vocabulary_size, embed_size)
encoder_embed_outputs = encoder_embed_layer(encoder_inputs)
encoder_lstm_layer = LSTM(embed_size, return_state=True)
_, encoder_state_h, encoder_state_c = encoder_lstm_layer(encoder_embed_outputs)
encoder_states = [encoder_state_h, encoder_state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embed_layer = Embedding(output_vocabulary_size, embed_size)
decoder_embed_outputs = decoder_embed_layer(decoder_inputs)
decoder_lstm_layer = LSTM(embed_size, return_sequences=True, return_state=True)
decoder_lstm_outputs, decoder_state_h, decoder_state_c = decoder_lstm_layer(decoder_embed_outputs, initial_state=encoder_states)
decoder_dense_layer = Dense(output_vocabulary_size, activation='softmax')
decoder_outputs = decoder_dense_layer(decoder_lstm_outputs)


# Define the model that will turn
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.summary()

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_steps = len(X_train) // BATCH_SIZE
validation_steps = len(X_test) // BATCH_SIZE
n_epochs = 100

weight_path="{}_weights.best.hdf5".format('english_french')
checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

model.fit_generator(batch_generator(X_train, y_train), steps_per_epoch=train_steps, epochs=n_epochs, validation_data=batch_generator(X_test, y_test), validation_steps=validation_steps, verbose=1, callbacks=[checkpoint])

model.load_weights(weight_path)
model.save('english_french_{}.h5'.format(VERSION))



# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state and a "start of sequence" token as target. Output will be the next target token
# 3) Repeat with the current target token and current states
model = models.load_model('english_french_{}.h5'.format(VERSION), compile=False)

decoder_embed_layer = model.layers[3]
decoder_lstm_layer = model.layers[5]
decoder_dense_layer = model.layers[6]


# Define sampling models
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
encoder_model.save('english_french_encoder_{}.h5'.format(VERSION))

decoder_state_input_h = Input(shape=(embed_size,))
decoder_state_input_c = Input(shape=(embed_size,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_embed_outputs = decoder_embed_layer(decoder_inputs)


decoder_lstm_outputs, decoder_state_h, decoder_state_c = decoder_lstm_layer(decoder_embed_outputs, initial_state=decoder_states_inputs)
decoder_states = [decoder_state_h, decoder_state_c]
decoder_outputs = decoder_dense_layer(decoder_lstm_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

encoder_model = models.load_model('english_french_encoder_{}.h5'.format(VERSION), compile=False)


def decode_sequence(input_text):

    input_text = [input_text]
    input_seq = np.array([input_text_to_x(text) for text in input_text])
    # Encode the input as state vectors.
    encoder_states_val = encoder_model.predict(input_seq) # shape = (2,1,256)

    # Generate empty target sequence of length 1.
    target_sentences = ["{"]
    target_seq = np.array([output_text_to_num(text) for text in target_sentences])

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    decoded_sentence_size = 0
    while not stop_condition:
        decoder_outputs_val, h, c = decoder_model.predict([target_seq] + encoder_states_val)

        # Sample a token
        sampled_token_index = np.argmax(decoder_outputs_val, axis=-1)[0,0]
        sampled_word = output_index_word_dict[sampled_token_index]
        decoded_sentence += sampled_word + ' '
        decoded_sentence_size += 1

        # Exit condition: either hit max length or find stop character.
        if sampled_word == '}' or decoded_sentence_size > output_max_size:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_sentences = [sampled_word]
        target_seq = np.array([output_text_to_num(text) for text in target_sentences])
        # Update states
        encoder_states_val = [h, c]

    return decoded_sentence


def decode_sequence2(input_text):
    input_text = [input_text]
    input_seq = np.array([input_text_to_x(text) for text in input_text])
    # Encode the input as state vectors.
    encoder_states_val = encoder_model.predict(input_seq)  # shape = (2,1,256)

    # Generate empty target sequence of length 1.
    target_sentences = ["{"]
    target_seq = np.array([output_text_to_num(text) for text in target_sentences])

    stop_condition = False
    decoded_sentence = ''
    decoded_sentence_size = 0
    while not stop_condition:
        decoder_outputs_val, h, c = decoder_model.predict([target_seq] + encoder_states_val)

        # Sample a token
        sampled_token_index = np.argmax(decoder_outputs_val, axis=-1)[0, -1]
        sampled_word = output_index_word_dict[sampled_token_index]
        decoded_sentence += sampled_word + ' '
        decoded_sentence_size += 1
        # Exit condition: either hit max length or find stop character.
        if sampled_word == '}' or decoded_sentence_size > output_max_size - 2:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_sentences = [target_sentences[0] + ' ' + sampled_word]
        target_seq = np.array([output_text_to_num(text) for text in target_sentences])
    return decoded_sentence



def tokenize_sentence(sentence):
    sentence = re.sub(r'[{}]', '', sentence).lower()
    return word_tokenize(sentence, language='french')

def calculate_bleu(reference, candidate):
    reference_list = tokenize_sentence(reference)
    candidate_list = tokenize_sentence(candidate)
    return sentence_bleu([reference_list], candidate_list, weights=(1, 0, 0, 0))

total_bleu = 0
input_sentences = []
for seq_index in range(2000,2100):
    # Take one sequence (part of the training set) for trying out decoding.
    input_sentence = input_texts[seq_index]
    decoded_sentence = decode_sequence(input_sentence)
    target_sentence = re.sub(r'[{}]', '', output_texts[seq_index]).lower()
    decoded_sentence = re.sub(r'[{}]', '', decoded_sentence).lower()
    bleu = calculate_bleu(target_sentence, decoded_sentence)
    total_bleu+=bleu
    if input_sentence not in input_sentences:
        input_sentences += [input_sentence]
        print('Input sentence: {}, Decoded sentence: {}'.format(input_sentence, decoded_sentence))
print(total_bleu/100) # 0.23 after 1 epoch # 0.50 after 10 epoch







