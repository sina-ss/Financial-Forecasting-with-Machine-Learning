import pandas as pd

class DataReader():
    def readCryptoData(name):
        df = pd.read_csv('Data\{}.csv'.format(name))
        return df
    
    