import pandas as pd

class DataReader():
    def readCryptoData():
        df = pd.read_csv('Data\XRPUSDT.csv')
        return df