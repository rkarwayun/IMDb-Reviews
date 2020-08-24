import pandas as pd
import numpy as np
import re
import os
import glob
import nltk

import tensorflow as tf

from keras import Sequential
from keras.layers import Input, Dense, Embedding, Dropout, Flatten, LSTM
from keras import regularizers
from keras.utils import np_utils, to_categorical
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.stem import WordNetLemmatizer 


# Download packages if necessary.
def setup():
	nltk.download('punkt')
	nltk.download('wordnet')



# This function reads each file as a line and removes all the special characters.
def readEachLine(agg, file_list):
	for fileName in file_list:
		f = open(fileName, 'r')
		for i in f:
			i = i.strip().lower()
			i = i.replace("<br />", "")
			i = re.sub(r"[^A-Za-z ]+",' ', i)
			# i = re.sub(r"[^A-Za-z0-9 ]+",' ', i)
			agg.append(i.strip().lower())
		f.close()
	return agg


# This function reads all the files.
def readFiles(flag):
	if(flag == 'train'):
		# parent = "assignment 3/data/aclImdb/train/"
		parent = "data/aclImdb/train/"
	else:
	    # parent = "assignment 3/data/aclImdb/test/"
		parent = "data/aclImdb/test/"

	pos_file_path = parent + "pos/*.txt"
	neg_file_path = parent + "neg/*.txt"

	# This lists all the files in the directly matching above name type.
	pos_file_list = glob.glob(pos_file_path)
	neg_file_list = glob.glob(neg_file_path)

	reviews = list()

	# Create dataset of reviews by reading positive and negative reviews.
	reviews = readEachLine(reviews, pos_file_list)
	reviews = readEachLine(reviews, neg_file_list)

	return reviews


# Read train or test input from the data, and return it along with the labels.
def getInput(flag):
	reviews_list = readFiles(flag)
	labels_list = [1 if i < 12500 else 0 for i in range(25000)]
	if flag != 'train':
		labels_list = to_categorical(labels_list)
		return np.array(reviews_list), labels_list
	else:
		# Incase of this function is called by training file, we also shuffle the training data.
		indices = [i for i in range(len(reviews_list))]
		np.random.seed(42)
		np.random.shuffle(indices)
		reviews_list_shuffled = list()
		labels_list_shuffled = list()
		for i in indices:
			reviews_list_shuffled.append(reviews_list[i])
			labels_list_shuffled.append(labels_list[i])
		labels_list_shuffled = to_categorical(labels_list_shuffled)
		return reviews_list_shuffled, labels_list_shuffled


# Remove stop words from the following stop word list.
def removeStopWords(all_reviews):
	stop_list = [ "the", "this", "that", "those", "their", "s", "t"]
	no_stop = list()
	for sent in all_reviews:
		temp = [i for i in sent if i not in stop_list]
		no_stop.append(temp)
	return no_stop


# Lemmatize tokens and return lemmatized list.
def lemmatizeList(all_tokens):
	lemma = list()
	lemmatizer = WordNetLemmatizer() 
	for sent in all_tokens:
		temp = [lemmatizer.lemmatize(i) for i in sent]
		lemma.append(temp)
	return lemma


# Tokenize and lemmatize the reviews.
def getTokens(all_reviews):
	setup()
	data_tokens = [nltk.word_tokenize(str(text)) for text in all_reviews]
	data_tokens_1 = removeStopWords(data_tokens)
	data_tokens_2 = lemmatizeList(data_tokens_1)
	maxi = 0
	for i in range(len(data_tokens_2)):
		if len(data_tokens_2[i]) > maxi:
			maxi = len(data_tokens_2[i])
	print(maxi)
	return data_tokens_2
