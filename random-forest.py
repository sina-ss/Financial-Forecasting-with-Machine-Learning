import pandas as pd
import pandas_datareader.data as pdr
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
import talib as ta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
import yfinance as yf
from data_loader import DataReader

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

def getPilotInput(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model = model.fit (X_train, y_train)
    probability = model.predict_proba(X_test)
    predicted = model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, predicted)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc_p = roc_auc_score(y_test, y_pred_proba)
    return fpr, tpr, auc_p

if __name__ == '__main__':
    df = DataReader.readCryptoData()
    n = int(input("Enter day range: "))
    needIndicator = int(input("Do You need Indicator? (0 for No and 1 for Yes!): "))
    X,y = getXY(df, n, needIndicator)
    X_train, X_test, y_train, y_test = getXYTrain(df, X, y)
    fpr_price, tpr_price, auc_price = getPilotInput(X_train, X_test, y_train, y_test)
    plt.subplot(2, 1, 1)
    plt.plot(fpr_price,tpr_price,label="data 1, auc="+str(auc_price))
    plt.legend(loc=4)

    X,y = getRY(df, n, needIndicator)
    X_train, X_test, y_train, y_test = getXYTrain(X, y)
    plt.subplot(2, 1, 2)
    fpr_return, tpr_return, auc_return = getPilotInput(X_train, X_test, y_train, y_test)
    plt.plot(fpr_return,tpr_return,color='r',label="'Bitcoin-RETURN', auc="+str(auc_return))
    plt.legend(loc=4)
    plt.show()
