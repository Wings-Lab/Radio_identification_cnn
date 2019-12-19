from keras.layers import Input, Dense, Embedding, concatenate, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Conv2D
from data_v2 import load_data
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.models import load_model
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
from numpy import argmax

saved_weights='cnn_model.h5'

print('Loading data')
x, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

if(os.path.isfile(saved_weights)):
	print("Loading previously trained weights")
	model = load_model(saved_weights)
else:
	print("Trained weights not found. Training the network")
	num_filters = 50
	drop = 0.5
	nb_epoch = 15

	#data preprocess, Reshape
	X_train = X_train.reshape(len(X_train), 2, 128, 1)
	X_test = X_test.reshape(len(X_test), 2, 128, 1)

	model = Sequential()
	model.add(Conv2D(num_filters, 2, activation='relu'))
	model.add(MaxPooling2D(pool_size=(1, 1), strides=None))
	model.add(Conv2D(num_filters, 1, activation='relu'))
	model.add(MaxPooling2D(pool_size=(1, 1), strides=None))
	model.add(Flatten())
	model.add(Dropout(drop))
	model.add(Dense(units=256, activation='relu'))
	model.add(Dense(units=4, activation='softmax'))

	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(X_train, y_train, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test))

results = model.predict(X_test, verbose=1)

test_size = len(results)
score = 0
for result, expected in zip(results, y_test):
	if argmax(result) == argmax(expected):
		score += 1
print("Score " + str(score))
print("Accuracy " + str((score*100)/test_size))
