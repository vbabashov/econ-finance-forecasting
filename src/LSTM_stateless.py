#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 15:20:54 2019

@author: vusalbabashov
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.optimizers import SGD
from math import sqrt
from matplotlib import pyplot
import numpy
import matplotlib.pyplot as plt
from numpy.random import seed
seed(0)
import tensorflow as tf
# tf.random.set_seed(0)


# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%d-%b-%Y')

#dataframe_columns = ['NEW.WIT.5', 'FIT.WIT.5', 'FIT.DEPO.5', 'NEW.WIT.10',
#       'FIT.WIT.10', 'FIT.DEPO.10', 'NEW.WIT.20', 'FIT.WIT.20', 'FIT.DEPO.20',
#       'NEW.WIT.50', 'FIT.WIT.50', 'FIT.DEPO.50', 'NEW.WIT.100', 'FIT.WIT.100',
#       'FIT.DEPO.100']



# Get the raw data values from the pandas data frame.
# denom='NEW.WIT.50'
# dataframe = pd.read_csv("/Users/vusalbabashov/Desktop/forecast/python_codes/02_Codes/calgary2019w43naive.csv", engine='python')
# data_raw = dataframe[denom]
# data_raw = data_raw.astype("float32")
# data_raw = data_raw.values
# data_raw = np.log1p(data_raw)

# Create a time series plot.
#plt.figure(figsize = (15, 5))
#plt.plot(data_raw, label = "Bank Note")
#plt.xlabel("Weeks")
#plt.ylabel("Pieces of Notes (thousands)")
#plt.title("Weekly Bank Note Demand for " + denom)
#plt.legend()
#plt.show()


def loadVB_daatset():
	denom='FIT.DEPO.5'
	dataframe = pd.read_csv("/Users/vusalbabashov/Desktop/forecast/python_codes/02_Codes/calgary2019w43naive.csv", engine='python')
	data_raw = dataframe[denom]
	data_raw = data_raw.astype("float32")
	data_raw = data_raw.values
	return data_raw, denom

def loadAD_dataset(region='R1', columnName='total'):
	filepath = "B:\\projects\\econ-finance-forecasting\\data"
	df = pd.read_csv(filepath+'\\'+region+'.csv')
	df.drop(['Region', 'Site'], inplace=True, axis=1)
	df.set_index(['Date'], inplace=True)
	df['total'] = df.sum(axis=1)
	data = df[columnName].astype("float32")
	return np.log1p(data.values), columnName

data_raw, denom = loadAD_dataset()

n_TSTEPS = 4
batch_SIZE = 64
n_NEURON=200
n_EPOCH = 300


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
# reshape from [samples, timesteps] into [samples, timesteps, features]
def fit_lstm(train, nbatch, nb_epoch, neurons, timesteps):
    X, y = train[ :, 0:-1], train[:, -1]
    #X = X.reshape(X.shape[0], 1, X.shape[1]) # use features
    X = X.reshape(X.shape[0], X.shape[1], 1)# use timesteps
    model = Sequential()
    model.add(LSTM(neurons, activation = "relu", input_shape=(X.shape[1], X.shape[2]), stateful = False, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer="adam", metrics=['accuracy'])
    history = model.fit(X, y, epochs=nb_epoch, batch_size=nbatch, verbose=0, validation_split = 0.2, shuffle = True)
    # list all data in history
    #print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss') 
    plt.ylabel('loss')
    plt.xlabel('epoch') 
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for accuracy
#    plt.plot(history.history['accuracy'])
#    plt.plot(history.history['val_accuracy'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'validation'], loc='upper left')
#    plt.show() 
    return model

# make a one-step forecast
def forecast_lstm(model, nbatch, X):
    #X = X.reshape(1, 1, len(X))  # use features
    X = X.reshape(1, len(X), 1)  # use timesteps
    yhat = model.predict(X, batch_size=nbatch)
    return yhat[0,0]



# transform data to be stationary
#raw_values = series_raw.values
diff_values = difference(data_raw, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, n_TSTEPS)
supervised_values = supervised.values

# Using 60% of data for training, 40% for validation.
TRAIN_SIZE = 0.8
train_size = int(len(data_raw) * TRAIN_SIZE)
test_size = len(data_raw) - train_size
print("Number of entries (training set, test set): " + str((train_size, test_size)))

#Split the data into train and test
train, test = supervised_values[0:-test_size], supervised_values[-test_size:]
#train, test = x_samples[0:train_size, 0:-1], x_samples[train_size:train_size + val_size, 0:-1]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

## fit the model
#lstm_model = fit_lstm(train_scaled, batch_SIZE, n_EPOCH, n_NEURON, n_TSTEPS)
#train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1) # 1 timestep 1 feature
##train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 1, 4) # 1 timestep 4 features
##train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), 4, 1) # 4 timestep 1 features
#
## forecast the entire training dataset to build up state for forecasting
#lstm_model.predict(train_reshaped, batch_SIZE)
#
## walk-forward validation on the test data
#predictions = list()
#for i in range(len(test_scaled)):
#    # make one-step forecast
#    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#    yhat = forecast_lstm(lstm_model, batch_SIZE, X)
#    #yhat = forecast_lstm(lstm_model, 1, X)
#    # invert scaling
#    yhat = invert_scale(scaler, X, yhat)
#    # invert differencing
#    yhat = inverse_difference(data_raw, yhat, len(test_scaled)+1-i)
#    # store forecast
#    predictions.append(yhat)
#   # expected = data_raw[len(train) + i + 1]
#   # print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
#
## report performance
#rmse = sqrt(mean_squared_error(data_raw[-test_size:], predictions))
#print('Test RMSE: %.3f' % rmse)


# repeat experiment
repeats = 10
error_scores = list()
for r in range(repeats):    
    # fit the model
    lstm_model = fit_lstm(train_scaled, batch_SIZE, n_EPOCH, n_NEURON, n_TSTEPS)
    # forecast the entire training dataset to build up state for forecasting
    #train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), n_TSTEPS, 1)
    train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), n_TSTEPS, 1) # 4 timestep 1 features
    lstm_model.predict(train_reshaped, batch_SIZE)    
    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(data_raw , yhat, len(test_scaled)+1-i)
        # invert the log scaling
        yhat = np.expm1(yhat)
        # store forecast
        predictions.append(yhat)    
    # report performance
    rmse = sqrt(mean_squared_error(data_raw[-test_size:], predictions))
    print('%d) Test RMSE: %.3f' % (r+1, rmse))
    error_scores.append(rmse)

# summarize results
results = DataFrame()
results['rmse'] = error_scores
print(results.describe())
results.boxplot()
pyplot.show()

