import pandas as pd
import numpy as np
# from pandas_datareader.data import DataReader
from sklearn.model_selection import train_test_split
# from eczema_model.config import config
# # from datetime import datetime
# from eczema_model.config.config import SYMBOLS, ROOT, DATA

print("hello world")

# def read_data(symbols="BTC-USD"):
#     ###Use if the data is read from online resources
#     # today = datetime.now()
#     # if(isinstance(symbols, str)):
#     #     data = DataReader(symbols, start="2014-15-09", end=today)
#     # else:
#     #     data = DataReader(f"{symbols}", start="2017-01-01", end=today)

#     # Line for already downloaded data

#     data = pd.read_csv(DATA,  na_values=np.nan).sort_values(by=["Date"])
#     data = data.dropna()
#     train = data.drop(data.tail(config.PREDICTIONWINDOW).index) 
#     valid = data.tail(7)
#     # print(X_test)

#     # train.to_csv(config.train, index=False)
#     # valid.to_csv(config.valid, index=False)
#     return train, valid

# if __name__ == "__main__":
#     df = read_data()
#     # print("Data splitting completed: ", df.head())