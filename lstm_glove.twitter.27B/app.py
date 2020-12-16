from flask import Flask, render_template, request, redirect, url_for

import os
import pathlib
import random
import datetime
import functools
import pickle
import re
import keras
import tweepy
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from flask import Flask
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from service import get_related_tweets
from service import get_related_reddit_comments
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
#text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+(?<![\w.-])@[A-Za-z][\w-]+"
MAX_NB_WORDS = 100000
MAX_SEQUENCE_LENGTH = 30
lab_to_sentiment = {0:"Negative", 4:"Positive"}
tokenizer = pickle.load( open( "tokenizer.pickle", "rb" ) )
countNeg=0
countPos=0
countLikelyNeg=0
countLikelyPos=0
countNeutral=0
length=0

countNegReddit=0
countPosReddit=0
countLikelyNegReddit=0
countLikelyPosReddit=0
countNeutralReddit=0
lengthReddit=0

def decode_sentiment(score):
	if(score>0.7):
		return "Positive"
	elif(score<0.3):
		return "Negative"
	elif(score>=0.3 and score<0.45):
		return "Likely Negative"
	elif(score<=0.7 and score>0.55):
		return "Likely Positive"
	elif(score>=0.45 and score<=0.55):
		return "Neutral"


def preprocessInput(text, stem=False):
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)

model = keras.models.load_model("my_model")

def requestResults(name):
	#userName = re.search("^https?://(www\.)?twitter\.com/(#!/)?(?<name>[^/]+)(/\w+)*$", userName)
	inputArray = get_related_tweets(name)
	input_df = pd.DataFrame(inputArray,columns=['text'])
	input_df.text = input_df.text.apply(lambda x: preprocessInput(x))
	inputList = np.array(input_df).flatten()
	input_df_padding = pad_sequences(tokenizer.texts_to_sequences(inputList),maxlen = MAX_SEQUENCE_LENGTH)
	input_df_scores = model.predict(input_df_padding, verbose=1, batch_size=10000)
	input_df_pred_1d = [decode_sentiment(score) for score in input_df_scores]
	global countNeg
	global countPos
	global countLikelyNeg
	global countLikelyPos
	global countNeutral
	global length
	for i in input_df_pred_1d:
		length = length+1
		if i=='Negative':
			countNeg=countNeg+1
		elif i=='Positive':
			countPos=countPos+1
		elif i=='Likely Negative':
			countLikelyNeg=countLikelyNeg+1
		elif i=='Likely Positive':
			countLikelyPos=countLikelyPos+1
		elif i=='Neutral':
			countNeutral=countNeutral+1
	return zip(inputArray, input_df_pred_1d)

def requestRedditResults(name):
	#userName = re.search(r'https://www.reddit.com/user/([^/?]+)', name).group(1)
	inputArray = get_related_reddit_comments(name)
	input_df = pd.DataFrame(inputArray,columns=['text'])
	input_df.text = input_df.text.apply(lambda x: preprocessInput(x))
	inputList = np.array(input_df).flatten()
	input_df_padding = pad_sequences(tokenizer.texts_to_sequences(inputList),maxlen = MAX_SEQUENCE_LENGTH)
	input_df_scores = model.predict(input_df_padding, verbose=1, batch_size=10000)
	input_df_pred_1d = [decode_sentiment(score) for score in input_df_scores]
	global countNegReddit
	global countPosReddit
	global countLikelyNegReddit
	global countLikelyPosReddit
	global countNeutralReddit
	global lengthReddit
	for i in input_df_pred_1d:
		lengthReddit = lengthReddit+1
		if i=='Negative':
			countNegReddit=countNegReddit+1
		elif i=='Positive':
			countPosReddit=countPosReddit+1
		elif i=='Likely Negative':
			countLikelyNegReddit=countLikelyNegReddit+1
		elif i=='Likely Positive':
			countLikelyPosReddit=countLikelyPosReddit+1
		elif i=='Neutral':
			countNeutralReddit=countNeutralReddit+1
	return zip(inputArray, input_df_pred_1d)

def getCount():
	return [countNeg,countPos]
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/', methods=['POST', 'GET'])
def get_data():
	if request.method == 'POST':
		twitter = re.search("^(?:https?:\/\/)?(?:www.)?(?:twitter.com)?[\/@]?([^\/?\s]+)",request.form['twitter']).group(1) 
		reddit = reddit = re.search("^(?:https?:\/\/)?(?:www.)?(?:reddit.com\/r)?[\/]?([^\/?\s]+)",request.form['reddit']).group(1)
		return redirect(url_for('success', twitter=twitter, reddit=reddit))
@app.route('/success/<twitter>/<reddit>',methods=['POST', 'GET']) 
def success(twitter, reddit):
	global countNeg
	global countPos
	global countLikelyNeg
	global countLikelyPos
	global countNeutral
	global length
	global countNegReddit
	global countPosReddit
	global countLikelyNegReddit
	global countLikelyPosReddit
	global countNeutralReddit
	global lengthReddit
	data = requestResults(twitter)
	data2 = requestRedditResults(reddit)
	count = [countNeg,countPos,countLikelyNeg,countLikelyPos,countNeutral,length]
	countReddit = [countNegReddit,countPosReddit,countLikelyNegReddit,countLikelyPosReddit,countNeutralReddit,lengthReddit]
	length = 0
	countLikelyNeg=0
	countLikelyPos=0
	countNeutral=0
	countNeg = 0
	countPos = 0
	lengthReddit =0 
	countLikelyNegReddit=0
	countLikelyPosReddit=0
	countNeutralReddit=0
	countNegReddit = 0
	countPosReddit = 0
	return render_template('success.html', data=data,data2=data2, count=count, countReddit=countReddit)
    #return "<xmp>" + str(requestResults(name)) + " </xmp> "
if __name__ == '__main__' :
	#start_ngrok()
	app.run(debug=True)