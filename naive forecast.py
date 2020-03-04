#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:43:27 2019

@author: vusalbabashov
"""

# naive forecast strategies for the power usage dataset
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import datetime
from pandas import DataFrame
# fix random seed for reproducibility
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)


#dataframe_columns = ['NEW.WIT.5', 'FIT.WIT.5', 'FIT.DEPO.5', 'NEW.WIT.10',
#       'FIT.WIT.10', 'FIT.DEPO.10', 'NEW.WIT.20', 'FIT.WIT.20', 'FIT.DEPO.20',
#       'NEW.WIT.50', 'FIT.WIT.50', 'FIT.DEPO.50', 'NEW.WIT.100', 'FIT.WIT.100',
#       'FIT.DEPO.100']


bank_note='FIT.DEPO.50'
dataframe = pd.read_csv("/Users/vusalbabashov/Desktop/forecast/python_codes/02_Codes/calgary2019w43naive.csv", engine='python')
data = dataframe[bank_note].values

# split a univariate dataset into train/test sets
def split_dataset(data):
    # split into train and test sets
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:len(data)]
    return train, test


train, test = split_dataset(data)



########################## last Week###########################
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.figure(figsize = (15, 5))
pyplot.plot(test, label = "Actual")
pyplot.plot(predictions, label = "Predicted")
pyplot.xlabel("Weeks")
pyplot.ylabel("Pieces of Notes (thousands)")
pyplot.title("Naive Forecast Strategy: Last Week to Predict Next Week " + bank_note)
pyplot.legend()
pyplot.show()




########################## Naive Strategy One Year Ago This Week###########################
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-52])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.figure(figsize = (15, 5))
pyplot.plot(test, label = "Actual")
pyplot.plot(predictions, label = "Predicted")
pyplot.xlabel("Weeks")
pyplot.ylabel("Pieces of Notes (thousands)")
pyplot.title("Naive Forecast Strategy: One Year Ago This Week " + bank_note)
pyplot.legend()
pyplot.show()

########################## Naive Strategy Weighted Average for Last Four Years###########################

weights = [0.48, 0.24, 0.16, 0.12]
# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(weights[0]*history[-52*1] + weights[1]*history[-52*2] + weights[2]*history[-52*3] + weights[3]*history[-52*4])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# line plot of observed vs predicted
pyplot.figure(figsize = (15, 5))
pyplot.plot(test, label = "Actual")
pyplot.plot(predictions, label = "Predicted")
pyplot.xlabel("Weeks")
pyplot.ylabel("Pieces of Notes (thousands)")
pyplot.title("Weighted Average Forecast Strategy " + bank_note)
pyplot.legend()
pyplot.show()
























## split a univariate dataset into train/test sets
#def split_dataset(data):
#	# split into standard weeks
#	train, test = data[1:-328], data[-328:-6]
#	# restructure into windows of weekly data
#	train = array(split(train, len(train)/7))
#	test = array(split(test, len(test)/7))
#	return train, test
#
## evaluate one or more weekly forecasts against expected values
#def evaluate_forecasts(actual, predicted):
#	scores = list()
#	# calculate an RMSE score for each day
#	for i in range(actual.shape[1]):
#		# calculate mse
#		mse = mean_squared_error(actual[:, i], predicted[:, i])
#		# calculate rmse
#		rmse = sqrt(mse)
#		# store
#		scores.append(rmse)
#	# calculate overall RMSE
#	s = 0
#	for row in range(actual.shape[0]):
#		for col in range(actual.shape[1]):
#			s += (actual[row, col] - predicted[row, col])**2
#	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
#	return score, scores
#
## summarize scores
#def summarize_scores(name, score, scores):
#	s_scores = ', '.join(['%.1f' % s for s in scores])
#	print('%s: [%.3f] %s' % (name, score, s_scores))
#
## evaluate a single model
#def evaluate_model(model_func, train, test):
#	# history is a list of weekly data
#	history = [x for x in train]
#	# walk-forward validation over each week
#	predictions = list()
#	for i in range(len(test)):
#		# predict the week
#		yhat_sequence = model_func(history)
#		# store the predictions
#		predictions.append(yhat_sequence)
#		# get real observation and add to history for predicting the next week
#		history.append(test[i, :])
#	predictions = array(predictions)
#	# evaluate predictions days for each week
#	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
#	return score, scores
#
#
## weekly persistence model
#def weekly_persistence(history):
#	# get the data for the prior week
#	last_week = history[-1]
#	return last_week[:, 0]
#
## week one year ago persistence model
#def week_one_year_ago_persistence(history):
#	# get the data for the prior week
#	last_week = history[-52]
#	return last_week[:, 0]
#
## load the new file
#dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
## split into train and test
#train, test = split_dataset(dataset.values)
## define the names and functions for the models we wish to evaluate
#models = dict()
#models['daily'] = daily_persistence
#models['weekly'] = weekly_persistence
#models['week-oya'] = week_one_year_ago_persistence
## evaluate each model
#days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
#for name, func in models.items():
#	# evaluate and get scores
#	score, scores = evaluate_model(func, train, test)
#	# summarize scores
#	summarize_scores(name, score, scores)
#	# plot scores
#	pyplot.plot(days, scores, marker='o', label=name)
## show plot
#pyplot.legend()
#pyplot.show()