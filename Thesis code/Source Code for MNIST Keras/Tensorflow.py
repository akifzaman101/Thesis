from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras import backend as K, optimizers

# Set seed for reproducibility
np.random.seed(1337)

# Define parameters
batch_size = 128
num_classes = 10
num_epochs = 16
img_rows, img_cols = 28, 28
num_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
input_shape = (1, img_rows, img_cols) if K.image_dim_ordering() == 'th' else (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], *input_shape).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], *input_shape).astype('float32') / 255
Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

# Build the model
model = Sequential([
    Conv2D(num_filters, kernel_size, padding='same', input_shape=input_shape),
    Activation('relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(48, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    Flatten(),
    Dense(96),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_data=(X_test, Y_test))

# Evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
