# import required packages
import pandas as pd
import numpy as np
import re
import os
import glob
import nltk
import pickle

import tensorflow as tf
from gensim.models import Word2Vec

from keras import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Flatten, LSTM
from keras import regularizers
from keras.utils import np_utils, to_categorical
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from utils import getInput, getTokens

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
	# 1. Load your saved model
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

	maxSentLen = 295	# 80 percentile sentence length in training dataset.

	model = load_model("models/20829490_NLP_model.model")

	# 2. Load your testing data


	test_raw, labels = getInput('test')

	# Tokenizing, removing stop words and lemmetizing.
	tokens1 = getTokens(test_raw)

	# Truncating longer sentences to 80 percentile sentence length.
	truncatedData = [' '.join(seq[:maxSentLen]) for seq in tokens1]


	# Processing Test Data.
	tokenizer = pickle.load(open("data/token.p", "rb"))
	final_data = tokenizer.texts_to_sequences(truncatedData)


	# Padding the data.
	final_data = pad_sequences(final_data, maxlen=maxSentLen, padding='post')

	# 3. Run prediction on the test data and print the test accuracy

	evalu = model.evaluate(final_data, labels)

	print("Test Accuracy and Loss is: ", str(evalu[1] * 100), "% and ", evalu[0], " respectively.", sep = '')

