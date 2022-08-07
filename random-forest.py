import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
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
    with open('Data\CryptoName.txt') as f:
        crypto_names = f.readline().split(',')
        for name in crypto_names:
            df = DataReader.readCryptoData(name)
            n = [2, 3, 5]
            needIndicator = [0, 1]
            print("====== name: {} ======".format(name))
            for i in n:
                for j in needIndicator:
                    print("n = {} and has indicator? {}".format(i,j))
                    X,y = getXY(df, i, j)
                    X_train, X_test, y_train, y_test = getXYTrain(df, X, y)
                    fpr_price, tpr_price, auc_price = getPilotInput(X_train, X_test, y_train, y_test)
                    print("price auc: {}".format(auc_price))
                    X,y = getRY(df, i, j)
                    X_train, X_test, y_train, y_test = getXYTrain(df, X, y)
                    fpr_return, tpr_return, auc_return = getPilotInput(X_train, X_test, y_train, y_test)
                    print("return auc: {}".format(auc_return))
            print("===================================")
