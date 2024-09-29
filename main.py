import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import time
import sys
import tkinter as tk
from tkinter import scrolledtext
from tkinter import PhotoImage
import urllib.request


'''step 1: Data aquisation'''
faqs = """The cat watched the raindrops race down the window.
A distant train whistle echoed through the quiet night.
She found a forgotten letter tucked between the pages of an old book.
The mountains glowed pink at dawn, heralding a new day.
Laughter erupted from the children playing in the park.
An old man sat on the bench, lost in memories of his youth.
The aroma of freshly baked bread filled the air.
Stars twinkled like diamonds scattered across the velvet sky.
A curious fox peeked out from behind the trees.
The waves crashed against the shore, whispering secrets to the sand.
She painted her dreams in vibrant colors on canvas.
A lone balloon floated away, drifting toward the horizon.
He discovered a hidden garden, overgrown but beautiful.
The clock chimed, marking the passage of time.
A gentle breeze rustled the leaves, singing a soft tune.
The library held stories waiting to be discovered.
A child's drawing of a monster brought laughter to the room.
Clouds danced across the sky, shifting shapes and stories.
The first snowflakes of winter landed softly on the ground.
An adventure awaited just beyond the horizon.
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])


'''step 2: data cleaning'''
input_sequences = []
for sentence in faqs.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i+1])

'''step 3: Data preprocessing'''
max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]

y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

'''step 4: modelling'''
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 100))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100)