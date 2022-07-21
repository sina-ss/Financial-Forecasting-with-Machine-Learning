import pandas as pd
import pandas_datareader.data as pdr
from pandas_datareader import data as pdr

import numpy as np

import datetime as dt

import talib as ta

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

import yfinance as yf

from data_loader import DataReader

import seaborn as sns


pd.options.mode.chained_assignment = None  # default='warn'

yf.pdr_override()

df = DataReader.readCryptoData()
X = df[['x1', 'x2', 'x3', 'x4', 'x5']]
y = df[['y']]

split = int(0.75*len(df))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
model = LogisticRegression()
model = model.fit (X_train,y_train)
#pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))

probability = model.predict_proba(X_test)

predicted = model.predict(X_test)

cnf_matrix = confusion_matrix(y_test, predicted)
cnf_matrix

#class_names=[0,1] # name  of classes
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(class_names))
#plt.xticks(tick_marks, class_names)
#plt.yticks(tick_marks, class_names)
# create heatmap
#sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
#ax.xaxis.set_label_position("top")
#plt.tight_layout()
#plt.title('Confusion matrix', y=1.1)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')

cross_val = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)

y_pred_proba = model.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()