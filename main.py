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

def generate_text():
    input_text = input_box.get("1.0", tk.END).strip()
    if not input_text:
        output_text.insert(tk.END, "Please enter some text to generate.")
        return
    for _ in range(10):
        token_text = tokenizer.texts_to_sequences([input_text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=max_len-1, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                input_text = input_text + " " + word
                output_text.insert(tk.END, word + " ")
                output_text.update()  
                break
        time.sleep(2)

'''step 5: User interface'''
def reset_text():
    input_box.delete("1.0", tk.END)
    output_text.delete("1.0", tk.END)
window = tk.Tk()
window.title("Text Generation Model")

window.option_add("*Font", "Mukta")

bg_image_url = "https://www.microsoft.com/en-us/research/uploads/prod/2023/03/AI_Microsoft_Research_Header_1920x720.png"
with urllib.request.urlopen(bg_image_url) as url:
    bg_image_data = url.read()

bg_photo = PhotoImage(data=bg_image_data)

window_width = window.winfo_screenwidth()
window_height = window.winfo_screenheight()

bg_photo_resized = bg_photo.subsample(int(bg_photo.width() / window_width), int(bg_photo.height() / window_height))

background_label = tk.Label(window, image=bg_photo_resized)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

input_frame = tk.Frame(window)
input_frame.pack(pady=10)

input_label = tk.Label(input_frame, text="Initialize text you want to generate")
input_label.pack()

input_box = scrolledtext.ScrolledText(input_frame, width=100, height=10)
input_box.pack()

button_frame = tk.Frame(window)
button_frame.pack(pady=5)

generate_button = tk.Button(button_frame, text="Generate", command=generate_text)
generate_button.pack(side=tk.LEFT, padx=5)

reset_button = tk.Button(button_frame, text="Reset", command=reset_text)
reset_button.pack(side=tk.LEFT, padx=5)

output_frame = tk.Frame(window)
output_frame.pack(pady=10)

output_label = tk.Label(output_frame, text="Generated text:")
output_label.pack()

output_text = scrolledtext.ScrolledText(output_frame, width=50, height=10)
output_text.pack()

window.mainloop()
