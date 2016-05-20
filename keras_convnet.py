'''
This file implements a Convolutional Neural Network 
using the Keras library on data consisting of handwritten digits.
The MNIST dataset was augmented with data from the Semeion dataset available
at the UCI ML repository. If the extended dataset is unavailable, the default
MNIST dataset can be loaded from the keras.datasets module

MNIST data: http://yann.lecun.com/exdb/mnist/
Semeion data: https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
'''


from __future__ import print_function
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(42)

#Read in extended training data and reshape
#Skip if extended data unavailable
temp_x_train = pd.read_csv("train_x.csv")
xtrain_mat = temp_x_train.as_matrix()
temp_y_train = np.genfromtxt("train_y.csv")

#Obtain MNIST test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = xtrain_mat.reshape(xtrain_mat.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = temp_y_train.astype('float32')
X_train /= 255
X_test /= 255

#Binarize the image
X_test[X_test > 0] = 1

X_train[X_train > 0] = 1

print('Input shape:', X_train.shape)
print(X_train.shape[0], 'training samples')
print(X_test.shape[0], 'testing samples')

#Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#Start sequential model
model = Sequential()

#Convolution 1: 32 filters (feature maps), 3x3 kernel
model.add(Convolution2D(nb_filters, 3, 3,
                        border_mode='valid',
                        input_shape=(1, 28, 28)))

#Activation 1
model.add(Activation('relu'))

#Convolution 2: 64 filters (feature maps), 3x3 kernel
model.add(Convolution2D(64, 3, 3))

#Activation 2
model.add(Activation('relu'))

#2x2 MaxPool (Downsampling)
# 64x24x24 -> 64x12x12
model.add(MaxPooling2D(pool_size=(2, 2)))

#Dropout 1: Avoid Overfitting
model.add(Dropout(0.25))

#Flatten: 64x12x12 = 9216
model.add(Flatten())

#Fully Connected NN 1: 9216 -> 128
model.add(Dense(128))

#Activation 4
model.add(Activation('relu'))

#Dropout 2
model.add(Dropout(0.5))

#Fully Connected NN 2: 128 -> 10
model.add(Dense(10))

#Softmax Regression: Gives Probabilities
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, nb_epoch=12,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_pred = model.predict_classes(X_test, verbose=0)

print(classification_report(y_pred, y_test))

#Save Model

keras_string = model.to_json()
open('keras_architecture.json', 'w').write(keras_string)
model.save_weights('keras_weights.h5')

'''
Instructions for loading saved model:

from keras.models import model_from_json

keras = model_from_json(open('keras_arch.json').read())
keras.load_weights('keras_weights.h5')

Compile using parameters of choice:

keras.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
'''