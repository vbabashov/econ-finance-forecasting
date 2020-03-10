#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:59:06 2019

@author: vusalbabashov
"""
#https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# LSTM for international airline passengers problem with window regression framing

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd 
import math
import numpy as np
import tensorflow
from math import log
from math import exp
from scipy.stats import boxcox
from math import sqrt
from pandas import Series
from pandas import concat
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import datetime
from pandas import DataFrame
from matplotlib import pyplot
# fix random seed for reproducibility
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)


#dataframe_columns = ['NEW.WIT.5', 'FIT.WIT.5', 'FIT.DEPO.5', 'NEW.WIT.10',
#       'FIT.WIT.10', 'FIT.DEPO.10', 'NEW.WIT.20', 'FIT.WIT.20', 'FIT.DEPO.20',
#       'NEW.WIT.50', 'FIT.WIT.50', 'FIT.DEPO.50', 'NEW.WIT.100', 'FIT.WIT.100',
#       'FIT.DEPO.100']




# Get the raw data values from the pandas data frame.
denom='FIT.DEPO.5'
dataframe = pd.read_csv("/Users/vusalbabashov/Desktop/forecast/python_codes/02_Codes/calgary2019w43naive.csv", engine='python')
data_raw = dataframe[denom]
data_raw = data_raw.astype("float32")
data_raw = data_raw.values
#data_raw = np.log1p(data_raw)

# Create a time series plot.
#plt.figure(figsize = (15, 5))
#plt.plot(data_raw, label = "Bank Note")
#plt.xlabel("Weeks")
#plt.ylabel("Pieces of Notes (thousands)")
#plt.title("Weekly Bank Note Demand for " + denom)
#plt.legend()
#plt.show()



n_STEPS = 1 # 4
batch_SIZE = 128  #70
n_NEURON=200
n_EPOCH = 500


# invert a boxcox transform for one value
def invert_boxcox(value, lam):
	# log case
	if lam == 0:
		return exp(value)
	# all other cases
	return exp(log(lam * value + 1) / lam)


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

# scale train and test data
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = MinMaxScaler(feature_range=(-1, 1))
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

###########  [samples, features]  ###############
# fit an LSTM network to training data
def fit_mlp(train, batch, nb_epoch, neurons):  
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0],X.shape[1])      #samples = X.shape[0], features = X.shape[1]
    model = Sequential()    
    model.add(Dense(neurons, activation = 'relu', input_dim=n_STEPS))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", optimizer = "adam", metrics=['accuracy'])
    history = model.fit(X, y, epochs = nb_epoch, batch_size = batch, verbose = 0, validation_split=0.2, shuffle = False)
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
def forecast_mlp(model, batch, X):
    X = X.reshape(1, len(X))
    yhat = model.predict(X, batch_size=batch)
    return yhat[0,0]



# power (box-cox) transform
#transformed, lmbda = boxcox(data_raw)
#print(transformed, lmbda)

# differencing to remove trend
#diff_values = difference(data_raw, 1)
#diff_values = difference(transformed, 1)

diff_values = data_raw
# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, n_STEPS)
supervised_values = supervised.values



TRAIN_SIZE = 0.8
train_size = int(len(data_raw) * TRAIN_SIZE)
test_size = len(data_raw) - train_size
print("Number of entries (training set, test set): " + str((train_size, test_size)))
#Split the data into train and test
train, test = supervised_values[0:-test_size], supervised_values[-test_size:]
# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)


# fit the model
mlp_model = fit_mlp(train_scaled, batch_SIZE, n_EPOCH, n_NEURON)
#reshaping for the training data prediction purposes
train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), n_STEPS)
#forecast the entire training dataset to build up state for forecasting
train_predict=mlp_model.predict(train_reshaped, batch_size=batch_SIZE)


# walk-forward validation on the test data
test_predict = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_mlp(mlp_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    #yhat = inverse_difference(data_raw, yhat, len(test_scaled)+1-i)
    # back-transform from log scale to the original scale
    #yhat = np.expm1(yhat)
    #inverted_yhat = invert_boxcox(yhat, lmbda)
    #print(inverted_yhat)
    # store forecast
    test_predict.append(yhat)
    expected = data_raw[len(train) + i]
    print('Week=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
# report performance
rmse = sqrt(mean_squared_error(data_raw[-test_size:], test_predict))
print('Test RMSE: %.3f' % rmse)




#repeats = 10
#error_scores = list()
#for r in range(repeats):    
#    # fit the model
#    mlp_model = fit_mlp(train_scaled, batch_SIZE, n_EPOCH, n_NEURON)
#    # forecast the entire training dataset to build up state for forecasting
#    train_reshaped = train_scaled[:, 0:-1].reshape(len(train_scaled), n_STEPS)
#    train_predict = mlp_model.predict(train_reshaped, batch_size=batch_SIZE)
#    # walk-forward validation on the test data
#    test_predict = list()
#    
#    for i in range(len(test_scaled)):
#        # make one-step forecast
#        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
#        yhat = forecast_mlp(mlp_model, 1, X)
#        # invert scaling
#        yhat = invert_scale(scaler, X, yhat)
#        # invert differencing
#        yhat = inverse_difference(data_raw, yhat, len(test_scaled)+1-i)
#        # back-transform from log scale to the original scale
#        yhat = np.expm1(yhat)
#        # store forecast
#        test_predict.append(yhat)
#        #expected = data_raw[len(train) + i + 1]
#        #print('Week=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
#   
#    # report performance
#    rmse = sqrt(mean_squared_error(data_raw[-test_size:], test_predict))
#    print('Test RMSE: %.3f' % rmse)
#    error_scores.append(rmse)        
#
## summarize results
#results = DataFrame()
#results['rmse'] = error_scores
#print(results.describe())
#results.boxplot()
#pyplot.show()







#Start with training predictions.
#a=np.array(data_raw)
#a.reshape(data_raw.shape[0],1)
#
#
#train_predict_plot = np.empty_like(data_raw)
#train_predict_plot[:, :] = np.nan
#train_predict_plot[n_STEPS:len(train_predict) + n_STEPS :] = train_predict
#
## Add test predictions.
#test_predict_plot = np.empty_like(data_raw)
#test_predict_plot[:, :] = np.nan
#test_predict_plot[len(train_predict) + (n_STEPS * 2) + 1 : len(data_raw) - 1, :] = test_predict
##
##
##print("============= Entire Dataset ====================")
### line plot of observed vs predicted on all data
#plt.figure(figsize = (15, 5))
#plt.plot(data_raw, label = "True value")
#plt.plot(train_predict_plot, label = "Train Predicted")
#plt.plot(test_predict_plot, label = "Test Predicted")
#plt.xlabel("Weeks")
#plt.ylabel("Pieces of Notes (Thousands)")
#plt.title("Comparison Actual vs. Predicted training / test for " + denom)
#plt.legend()
#plt.show()
#
#
#
##print("============= Test Dataset ====================")
## line plot of observed vs predicted on all data
#plt.figure(figsize = (15, 5))
#plt.plot(data_raw[-test_size:], label = "Raw Data")
#plt.plot(test_predict, label = "Test Predicted")
#plt.xlabel("Weeks")
#plt.ylabel("Pieces of Notes (Thousands)")
#plt.title("Actual vs. Predicted test for " + denom)
#plt.legend()
#plt.show()









## convert an array of values into a dataset matrix
#def create_dataset(dataset, look_back=1):
#    dataX, dataY = [], []
#    for i in range(len(dataset)-look_back-1):
#        a = dataset[i:(i+look_back), 0]
#        dataX.append(a)
#        dataY.append(dataset[i + look_back, 0])
#    return numpy.array(dataX), numpy.array(dataY)
#
#
#batch_size = 120
## normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(data_raw)
## split into train and test sets
#train_size = int(len(dataset) * 0.85)
#test_size = len(dataset) - train_size
##test_size = 16
##train_size = len(dataset) - test_size
#train, test = dataset[:train_size,:], dataset[train_size:len(dataset),:]
## reshape into X=t and Y=t+1
#look_back = 2
#trainX, trainY = create_dataset(train, look_back)
#testX, testY = create_dataset(test, look_back)
##create MLP model
#model = Sequential()    
#model.add(Dense(50, activation = 'relu', input_dim=look_back))
##model.add(Dense(50, activation = 'relu'))
#model.add(Dense(1))
#model.compile(loss = "mean_squared_error", optimizer = "adam")
#history= model.fit(trainX,trainY, epochs = 300, batch_size = batch_size , verbose = 0, validation_split=0.15, shuffle = False)
## list all data in history
#print(history.history.keys())
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
## make predictions
#trainPredict = model.predict(trainX)
#testPredict = model.predict(testX)
## invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
## shift train predictions for plotting
#trainPredictPlot = numpy.empty_like(dataset)
#trainPredictPlot[:, :] = numpy.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
## shift test predictions for plotting
#testPredictPlot = numpy.empty_like(dataset)
#testPredictPlot[:, :] = numpy.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
## plot baseline and predictions
##plt.plot(scaler.inverse_transform(dataset))
##plt.plot(trainPredictPlot)
##plt.plot(testPredictPlot)
##plt.show()
##
#
## Create the plot.
#plt.figure(figsize = (15, 5))
#plt.plot(scaler.inverse_transform(dataset), label = "True value")
#plt.plot(trainPredictPlot, label = "Training set prediction")
#plt.plot(testPredictPlot, label = "Test set prediction")
#plt.xlabel("Weeks")
#plt.ylabel("Pieces of Notes (thousands)")
#plt.title("Comparison true vs. predicted training / test for " + denom )
#plt.legend()
#plt.show()




# =============================================================================
# ###############LSTM for Regression with Time Steps###################################
# # LSTM for international airline passengers problem with time step regression framing
# 
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_shape=(look_back, 1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# # make predictions
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
# 
# 
# ################LSTM with Memory Between Batches###############3333
# 
# # LSTM for international airline passengers problem with memory
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# # reshape into X=t and Y=t+1
# look_back = 3
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# # create and fit the LSTM network
# batch_size = 1
# model = Sequential()
# model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# for i in range(100):
#     model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
#     model.reset_states()
# # make predictions
# trainPredict = model.predict(trainX, batch_size=batch_size)
# model.reset_states()
# testPredict = model.predict(testX, batch_size=batch_size)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
# 
# 
# 
# #######################Stacked LSTMs with Memory Between Batches################
# # Stacked LSTM for international airline passengers problem with memory
# 
# # fix random seed for reproducibility
# numpy.random.seed(7)
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(-1, 1))
# dataset = scaler.fit_transform(dataset)
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# # reshape into X=t and Y=t+1
# look_back = 4
# trainX, trainY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# # create and fit the LSTM network
# batch_size = 1
# model = Sequential()
# model.add(LSTM(128, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
# model.add(LSTM(32, batch_input_shape=(batch_size, look_back, 1), stateful=True))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# for i in range(100):
#     model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
#     model.reset_states()
# # make predictions
# trainPredict = model.predict(trainX, batch_size=batch_size)
# model.reset_states()
# testPredict = model.predict(testX, batch_size=batch_size)
# # invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
# 
# =============================================================================
