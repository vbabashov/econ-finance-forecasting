# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, datetime, Series, read_csv, read_excel, concat
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from numpy.random import seed
from math import sqrt
import matplotlib.pyplot as plt
from math import log, exp
from statsmodels.tsa import stattools



data = pd.read_csv("C:\\Users\\babv\\Desktop\\forecasting\\New_Data\\Ajit.csv", header = 0, parse_dates = ['Date'], index_col = ['Date'])
# to visualise al the columns in the dataframe
#pd.pandas.set_option('display.max_columns', None)
data.head()



# copy the dataframe and aggregate to the weekly level in each region
dataset = data.copy ()
dataset_by_region = dataset['2014-01-01':].groupby('Region').resample("W").sum().reset_index(level=0)
dataset_by_region



region_="R1"
denom_ = "5FD"
df_note = dataset_by_region.loc[dataset_by_region["Region"] == region_, [denom_]]



#Feature Engineering for Time Series #1: Date-Related Features
df_note['year'] = df_note.index.year 
df_note['month'] = df_note.index.month
df_note['week_num'] = df_note.index.week
df_note

#Feature Engineering for Time Series #2: Lag and Window Features
df_note ['lag1'] = df_note[denom_].shift(1)
df_note ['lag4'] = df_note[denom_].shift(4)
df_note ['lag52'] = df_note[denom_].shift(52)
#df_note ['diff1'] = df_note [denom_].diff(1)
#df_note ['diff52'] = df_note [denom_].diff(52)
df_note ['rolling_mean_4'] = df_note[denom_].rolling(window=4).mean()
#df_note ['rolling_mean_2v1'] = df_note[denom_].rolling(window=2).mean().shift()
df_note ['rolling_mean_52'] = df_note[denom_].rolling(window=52).mean()
df_note ['expanding_mean'] = df_note[denom_].expanding(2).mean()
#df_note ['min'] = df_note[denom_].expanding(2).min()
#df_note ['max'] = df_note[denom_].expanding(2).max()
#df_note ['log1p'] = np.log1p(df_note[denom_])
df_note

df_note = df_note.dropna()
df_note

features = ['lag1', 'lag4', 'lag52','rolling_mean_4', 'rolling_mean_52', 'expanding_mean'] 
targets = denom_
#targets = 'log1p'

df_note = df_note [features + [targets]]
df_note

#dataframe
#Train = df_note.loc['2016-01-04' : '2018-12-31', features]
#Test =  df_note.loc['2019-01-01' : , denom_]

train = df_note.loc['2016-01-04' : '2018-12-31'].values
test =  df_note.loc['2019-01-01' : ].values


##########################################################################################################
#                                                Some key hyperparamaters                                
##########################################################################################################


n_TSTEPS=1
batch_SIZE = test.shape[0] #70
n_NEURON_1 = train.shape[1]
n_NEURON_2 = train.shape[1]
n_NEURON_3 = train.shape[1]
n_EPOCH = 1500



# This function creates differenced time series. Degree of differencing can be specified, 1 for detrending, 52 for deaseasoning
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# This function inverts the differenced time series achieved via the difference function defined above
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
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]




###########  Fit a feed forward MLP model [samples, features]  ###############
def fit_mlp(train, batch, nb_epoch, n_neuron1, n_neuron2, n_neuron3):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], X.shape[1])      #samples = X.shape[0], features = X.shape[1]
    model = Sequential()
    model.add(Dense(n_neuron1, activation = 'relu', input_dim=X.shape[1]))
    model.add(Dense(n_neuron2))
    model.add(Dense(n_neuron3))
    #model.add(Dropout(0.4))
    model.add(Dense(1))
    opt = "adam"
    #optimizer=optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss = "mean_squared_error",  optimizer = opt, metrics=['accuracy'])  
    history = model.fit(X, y, epochs = nb_epoch, batch_size = batch, verbose = 0, validation_split=0.15, shuffle = False)
    # list all data in history
    #print(history.history.keys())
    # summarize history for loss
    plt.figure(1,figsize=(6,6),dpi=100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return model

# make a one-step forecast with MLP
def forecast_mlp(model, batch, X):
    X = X.reshape(1, len(X))
    yhat = model.predict(X, batch_size=batch)
    return yhat[0,0]





# fit an LSTM network to training data
# reshape from [samples, timesteps] into [samples, timesteps, features]
def fit_lstm(train, nbatch, nb_epoch,  n_neuron1, n_neuron2, n_neuron3, timesteps):
    X, y = train[ :, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], n_TSTEPS, X.shape[1])  # use features
    #X = X.reshape(X.shape[0], X.shape[1], 1) # use timesteps
    model = Sequential()
    model.add(LSTM(n_neuron1, activation = "relu", input_shape=(X.shape[1], X.shape[2]), stateful = False, return_sequences=True))
    model.add(LSTM(n_neuron2, stateful = False, return_sequences=True))
    model.add(LSTM(n_neuron3))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    opt = "adam"
    #optimizer=optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X, y, epochs=nb_epoch, batch_size=nbatch, verbose=0, validation_split = 0.15, shuffle = False)
    # list all data in history
    #print(history.history.keys())
    # summarize history for loss
    plt.figure(1,figsize=(6,6),dpi=100)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss') 
    plt.ylabel('loss')
    plt.xlabel('epoch') 
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    return model


# make a one-step forecast
def forecast_lstm(model, nbatch, X):
    X = X.reshape(1, 1, len(X))  # use features
    #X = X.reshape(1, len(X), 1)  # use timesteps
    yhat = model.predict(X, batch_size=nbatch)
    return yhat[0,0]




# differencing to remove trend
#diff_values = difference(data_raw, 1)

# transform the scale of the data
#scaler, train_scaled, test_scaled = scale(train, test)





#################################################################################################
                   #   MLP Model fitting and prediciton  
################################################################################################

# fit the model MLP
mlp_model = fit_mlp(train, batch_SIZE, n_EPOCH, n_NEURON_1, n_NEURON_2, n_NEURON_3)

#forecast the entire training dataset to build up state for forecasting
train_predict = mlp_model.predict(train[ :, 0:-1], batch_size=batch_SIZE)
#predict the test test
prediction_MLP = mlp_model.predict(test[ :, 0:-1], batch_size=batch_SIZE)

# report performance
rmse = sqrt(mean_squared_error(test[:,-1], prediction_MLP))
print('Test RMSE: %.2f' % rmse)

mae = mean_absolute_error(test[:,-1], prediction_MLP)
print('Test MAE: %.2f' % mae)




#################################################################################################
                   #   LSTM Model fitting and prediciton  
################################################################################################


# fit the model LSTM
lstm_model = fit_lstm(train, batch_SIZE, n_EPOCH, n_NEURON_1, n_NEURON_2, n_NEURON_3, n_TSTEPS)

# reshape the training data to samples, timesteps, features format
train_reshaped = train[:, 0:-1].reshape(train[ :, 0:-1].shape[0], n_TSTEPS, train[ :, 0:-1].shape[1]) 

# forecast the entire training dataset to build up state for forecasting
lstm_model.predict(train_reshaped, batch_SIZE)

# reshape the testing data to samples, timesteps, features format
test_reshaped = test[:, 0:-1].reshape(test[ :, 0:-1].shape[0], n_TSTEPS, test[ :, 0:-1].shape[1]) 
#predict the test test
prediction_LSTM = lstm_model.predict(test_reshaped, batch_size=batch_SIZE)

# report performance
rmse = sqrt(mean_squared_error(test[:,-1], prediction_LSTM))
print('Test RMSE: %.2f' % rmse)

mae = mean_absolute_error(test[:,-1], prediction_LSTM)
print('Test MAE: %.2f' % mae)
