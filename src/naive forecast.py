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


