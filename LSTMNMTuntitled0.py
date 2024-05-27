# -*- coding: utf-8 -*-
"""
Created on Wed May 15 02:51:05 2024

@author: Asus
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Example dataset
input_texts = ['hello', 'how are you', 'goodbye']
target_texts = ['hola', 'como estas', 'adios']
input_characters = []
[input_characters.extend(word) for word in input_texts]
input_characters = sorted(set(input_characters))
target_chars = sorted({char for word in target_texts for char in word})
encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_chars)
max_encoder_length = max(len(word) for word in input_texts)
max_decoder_length = max(len(word) for word in target_texts)

input_token = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_chars)])
reverse_target_index = dict((i, char) for char, i in target_token_index.items())


encoder_input_data = np.zeros((len(input_texts), max_encoder_length, encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(target_texts), max_decoder_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(target_texts), max_decoder_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None, encoder_tokens))
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=1,epochs=10,validation_split=0.2)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_hidden_input = Input(shape=(latent_dim,))
decoder_state = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_hidden_input, decoder_state]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    states_val = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_val)

        sample_token = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_index[sample_token]
        decoded_sentence += sampled_char

        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sample_token] = 1.

        states_val = [h, c]

    return decoded_sentence


target_chars = set()
for word in target_texts:
    for char in word + '\t':
        target_chars.add(char)

target_chars = sorted(target_chars)
target_token_index = {char: i for i, char in enumerate(target_chars)}

print("Target Characters:", target_chars)
print("Target Token Index:", target_token_index)

# Example input sequence
input_seq = encoder_input_data[0:1]  # Take the first input sequence for demonstration

# Decode the input sequence
decoded_sentence = decode_sequence(input_seq)

print("Input English phrase:", input_texts[0])
print("Translated Spanish phrase:", decoded_sentence)
