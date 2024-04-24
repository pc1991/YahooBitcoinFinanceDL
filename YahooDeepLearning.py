# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:47:41 2024

@author: pchri
"""

import yfinance as yf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#USD/EUR price levels for the past 10 years
symbol = 'BTC-USD'
raw = yf.download(symbol, start="2014-01-01", end="2024-04-21")['Adj Close']
data = pd.DataFrame(raw)

#computing returns and adding it to the DataFrame
data['return'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))

#if market direction >= 0 -> classify as 1, else classify as 0
data['direction'] = np.where(data['return'] > 0, 1, 0)

data.head()

#create 5 columns for each lag representing the past day's return
lags = 5

cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
    
data.dropna(inplace=True)

data.round(4).tail()

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import random

optimizer = Adam(learning_rate=0.0001)

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(100)
    
set_seeds()
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(lags,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

train, test = np.split(data, [int(.70*len(data))])

mu, std = train.mean(), train.std()

train_data = (train - mu) / std
test_data = (test - mu) / std

%%time
model.fit(train[cols],
          train['direction'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False)

res = pd.DataFrame(model.history.history)

#accuracy and validation of the model
res[['accuracy', 'val_accuracy']].plot(figsize=(10,6), style='--')

model.evaluate(train_data[cols], train['direction'])

#creating market prediction
pred = np.where(model.predict(train_data[cols]) > 0.5, 1, 0)
pred

pred[:30].flatten()

#transforming predictions to long-short
train['prediction'] = np.where(pred > 0, 1, -1)

#compute strategy returns given the positions
train['strategy'] = (train['prediction'] * train['return'])

train[['return', 'strategy']].sum().apply(np.exp)

train[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))

#add momentum, volatility, and distance to the frame
data['momentum'] = data['return'].rolling(5).mean().shift(1)
data['volatility'] = data['return'].rolling(20).std().shift(1)
data['distance'] = (data['Adj Close'] - data['Adj Close'].rolling(50).mean()).shift(1)

data.dropna(inplace=True)

cols.extend(['momentum', 'volatility', 'distance'])

#new dataframe
data.round(4).tail()

#refitting train/test model

train, test = np.split(data, [int(.70*len(data))])

mu, std = train.mean(), train.std()

train_data = (train - mu) / std
test_data = (test - mu) / std

#update Dense Layers to 32

set_seeds()
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(len(cols),)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

%%time
model.fit(train[cols],
          train['direction'],
          epochs=25, verbose=False,
          validation_split=0.2, shuffle=False)

model.evaluate(train_data[cols], train['direction'])

pred = np.where(model.predict(train_data[cols]) > 0.5, 1, 0)

train['prediction'] = np.where(pred > 0, 1, -1)

train['strategy'] = (train['prediction'] * train['return'])

train[['return', 'strategy']].sum().apply(np.exp)

train[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))

model.evaluate(test_data[cols], test['direction'])

pred = np.where(model.predict(test_data[cols]) > 0.5, 1, 0)

test['prediction'] = np.where(pred > 0, 1, -1)

test['prediction'].value_counts()

test['strategy'] = (test['prediction'] * test['return'])

test[['return', 'strategy']].sum().apply(np.exp)

test[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))