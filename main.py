import numpy as np
import time
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


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
