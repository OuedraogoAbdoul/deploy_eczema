from regression_model import pipeline as pipeline
from regression_model.config import config
from regression_model.get_data import read_data

import pandas as pd


def train_models():
    train, valid = read_data()
    train.reset_index()
    valid.reset_index()
    
    train = train.drop(["Date", "Volume"], axis=1)
    valid = valid.drop(["Date", "Volume"], axis=1)

    print(train.head())
    
    pipeline.btc_price_pipeline.fit(train, valid)

    return train

if __name__ == "__main__":
    df = train_models()
    # print("Data splitting completed: ", df.head())