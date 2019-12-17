from keras.layers import Input, Dense, Embedding, concatenate, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers.core import Reshape, Flatten
from keras.layers.convolutional import Conv2D
from data import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
from numpy import argmax

saved_weights='cnn_model.h5'

print('Loading data')
x, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sequence_length = 636

if(os.path.isfile(saved_weights)):
	print("Loading previously trained weights")
	model = load_model(saved_weights)
else:
	print("Trained weights not found. Training the network")
	embedding_dim = 256
	num_filters = 50
	drop = 0.5

	nb_epoch = 1
	batch_size = 128

	inputs = Input(shape=(sequence_length,), dtype='int32')
	embedding = Embedding(output_dim=embedding_dim, input_dim=sequence_length, input_length=sequence_length)(inputs)
	reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

	conv_0 = Conv2D(num_filters, (1, 3), activation='relu')(reshape)
	maxpool_0 = MaxPooling2D(pool_size=(2, 2), strides=None)(conv_0)

	conv_1 = Conv2D(num_filters, (2, 3), activation='relu')(maxpool_0)
	maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=None)(conv_1)

	flatten = Flatten()(maxpool_1)
	dropout = Dropout(drop)(flatten)
	output = Dense(units=636, activation='softmax')(dropout)

	# this creates a model that includes
	model = Model(inputs=inputs, outputs=output)
	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

	model.fit(X_train.T, y_train.T, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test.T, y_test.T))  # starts training
	model.save(saved_weights)

results = model.predict(X_test, verbose=1)

test_size = len(results)
score = 0
for result, expected in zip(results, y_test):
	if argmax(result) == argmax(expected):
		score += 1
print("Score " + str(score))
print("Accuracy " + str((score*100)/test_size))
