from tabnanny import verbose
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.optimizers import RMSprop
from data_loader import DataReader
import pandas as pd
import numpy as np
from keras import losses

def getXY(df ,n, needIndicator):
    if (needIndicator == 0):
        if (n == 2):
            X = df[['x1', 'x2']]
        if (n == 3):    
            X = df[['x1', 'x2', 'x3']]
        else:
            X = df[['x1', 'x2', 'x3', 'x4', 'x5']]
    else:
        if (n == 2):
            X = df[['x1', 'x2','14-day RSI','MA-2','EMA-2','14-day MFI']]
        if (n == 3):    
            X = df[['x1', 'x2', 'x3','14-day RSI','MA-2','EMA-2','14-day MFI']]
        else:
            X = df[['x1', 'x2', 'x3', 'x4', 'x5','14-day RSI','MA-2','EMA-2','14-day MFI']]
    y = df[['y']]
    y = y.iloc[:,-1:].values.ravel()
    return X,y    

def getRY(df ,n, needIndicator):
    if (needIndicator == 0):
        if (n == 2):
            X = df[['r1', 'r2']]
        if (n == 3):    
            X = df[['r1', 'r2', 'r3']]
        else:
            X = df[['r1', 'r2', 'r3', 'r4', 'r5']]
    else:
        if (n == 2):
            X = df[['r1', 'r2','14-day RSI','MA-2','EMA-2','14-day MFI']]
        if (n == 3):    
            X = df[['r1', 'r2', 'r3','14-day RSI','MA-2','EMA-2','14-day MFI']]
        else:
            X = df[['r1', 'r2', 'r3', 'r4', 'r5','14-day RSI','MA-2','EMA-2','14-day MFI']]
    y = df[['y']]
    y = y.iloc[:,-1:].values.ravel()
    return X,y    

def getXYTrain(df, X, y):
    split = int(0.75*len(df))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    return  X_train, X_test, y_train, y_test


def getModel():
    model = Sequential()
    model.add(Dense(64, input_dim=5, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=losses.BinaryCrossentropy(), optimizer =RMSprop(), metrics=['accuracy'])
    return model

if __name__ == '__main__':
    df = DataReader.readCryptoData()
    for i in range (len(df['y'])):
        if(df['y'][i] == -1):
            df['y'][i] += 1
    
    n = int(input("Enter day range: "))
    needIndicator = int(input("Do You need Indicator? (0 for No and 1 for Yes!): "))
    X,y = getXY(df, n, needIndicator)
    X_train, X_test, y_train, y_test = getXYTrain(df, X, y)
    model = getModel()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("loss-price:")
    print(loss)
    print("Accuracy-price:")
    print(accuracy)
    X,y = getRY(df, n, needIndicator)
    X_train, X_test, y_train, y_test = getXYTrain(X, y)
    model = getModel()
    model.fit(X_train, y_train, epochs=100, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test)
    print("loss-return:")
    print(loss)
    print("Accuracy-return:")
    print(accuracy)