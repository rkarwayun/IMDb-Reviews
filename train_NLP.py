# import required packages
import pandas as pd
import numpy as np
import re
import os
import glob
import nltk
import pickle

import tensorflow as tf

from keras import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D
from keras import regularizers
from keras.utils import np_utils, to_categorical
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils import getInput, getTokens, removeStopWords

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow




# Tokenize and vectorize the data.
def prepareData(truncatedData, percentile):

	tokenizer = Tokenizer(10000)
	
	tokenizer.fit_on_texts(truncatedData)
	final_data = tokenizer.texts_to_sequences(truncatedData)

	# Padding the data.
	final_data = pad_sequences(final_data, maxlen=percentile, padding='post')
	pickle.dump(tokenizer, open("data/token.p", "wb"))

	return final_data, tokenizer



# Creates and returns the model.
def returnModel():

	# Clear any previous model.
	tf.keras.backend.clear_session()

	# Defining a sequential model.
	model = Sequential()

	# Embedding layer.
	emb = Embedding(input_dim=vocabSize + 1, output_dim=opt_dim,
					input_length=percentile)
	model.add(emb)

	model.add(Conv1D(50, 3, padding='same', activation='relu'))
	model.add(MaxPooling1D())

	model.add(Dropout(0.25))

	model.add(Flatten())

	model.add(Dense(10, activation='tanh'))
	model.add(Dropout(0.25))

	model.add(Dense(units=2, activation='softmax'))

	model.summary()

	return model



def trainAndSaveModel():
	model = returnModel()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(final_data, labels, batch_size=500, epochs=3)

	model.save("models/20829490_NLP_model.model")

	evalu = model.evaluate(final_data, labels)

	print("Train Accuracy and Loss is: ", str(evalu[1] * 100), "% and ", evalu[0], " respectively.", sep = '')



if __name__ == "__main__": 
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

	# 3. Save your model
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	# Reading the training data.
	train_raw, labels = getInput('train')

	# Tokenizing the training data.
	tokens = getTokens(train_raw)
	# tokens = removeStopWords(tokens)

	opt_dim = 50

	
	# Finding the 80 percentile sentence length.
	# percentile = int(np.percentile([len(seq) for seq in tokens], 80))
	# print('80th Percentile Sentence Length:', percentile)
	percentile = 295	# 80th Percentile Sentence Length, Found using above two lines.

	# Truncate the data at 80 percentile sentence length.
	truncatedData = [' '.join(seq[:percentile]) for seq in tokens]

	# Vectorize the data.
	final_data, tok = prepareData(truncatedData, percentile)

	vocabSize = len(tok.word_index)
	print("Vocaublary size is:", vocabSize)

	# Train and save the model.
	trainAndSaveModel()
