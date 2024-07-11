import os
import math
import random

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data paths
train_data_dir = 'data/'
test_data_dir = 'data/'
# classes = ['Apple___healthy', 'Apple___rust', 'Bell_pepper___bacterial_spot', 
#            'Bell_pepper___healthy', 'Tomato___septoria_leaf_spot', 'Tomato___healthy',
#           'Grape___black_rot', 'Grape___healthy']
# Define image size and batch size
image_size = (28, 28)
batch_size = 32

# Use ImageDataGenerator to load and preprocess the data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,          # Normalize pixel values to [0, 1]
    shear_range=0.2,              # Apply shear transformation
    zoom_range=0.2,               # Apply zoom transformation
    horizontal_flip=True         # Flip images horizontally
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)  # Only rescale for test data

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(60, activation='softmax')  # Change number of units to 60
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=5, validation_data=test_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)

