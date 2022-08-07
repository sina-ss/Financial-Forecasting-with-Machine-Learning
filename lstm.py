from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from data_loader import DataReader
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Dropout , LSTM
from keras.optimizers import RMSprop

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

if __name__ == '__main__':
    with open('Data\CryptoName.txt') as f:
        crypto_names = f.readline().split(',')
        crypto_names = ['ADAUSDT']
        for name in crypto_names:
            df = DataReader.readCryptoData(name)
            n = [3, 5]
            needIndicator = [0, 1]
            print("====== name: {} ======".format(name))
            for i in n:
                for j in needIndicator:
                    print("n = {} and has indicator? {}".format(i,j))
                    X,y = getXY(df, i, j)
                    X_train, X_test, y_train, y_test = getXYTrain(df, X, y)
                    model = Sequential()
                    model.add(LSTM(64, return_sequences=True,input_shape=(i + j*4, 1),activation='relu'))
                    model.add(Dropout(0.2))
                    model.add(LSTM(32,activation='relu'))
                    model.add(Dropout(0.2))
                    model.add(Dense(1, activation='sigmoid'))
                    model.compile(loss=losses.BinaryCrossentropy(), optimizer =RMSprop(), metrics=['accuracy'])

                    earlystopping = callbacks.EarlyStopping(monitor ="accuracy", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)
                    
                    model.fit(X_train, y_train, batch_size = 128, epochs=9*(i + (4*j)), callbacks =[earlystopping])

                    loss, accuracy = model.evaluate(X_test, y_test)
                    print("loss-price:")
                    print(loss)
                    print("Accuracy-price:")
                    print(accuracy)

                    X,y = getRY(df, i, j)
                    X_train, X_test, y_train, y_test = getXYTrain(df, X, y)
                    model.fit(X_train, y_train, epochs=3*(i + (4*j)))
                    loss, accuracy = model.evaluate(X_test, y_test)
                    print("loss-return:")
                    print(loss)
                    print("Accuracy-return:")
                    print(accuracy)
            print("===================================")