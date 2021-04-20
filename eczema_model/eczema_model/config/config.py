# Import libraries 
from pathlib import Path

# Get main folder path
ROOT = f"{Path(__file__).resolve().parents[2]}"
ROOT = ROOT.split("/")[-1]

## Data path
SYMBOLS = "BTC-USD"
DATASETS = ROOT + "/data/"
train = DATASETS + "train.csv"
valid = DATASETS + "valid.csv"


DATAFILE = "BTC-USD.csv"
DATA = DATASETS + DATAFILE
WRONGDATAPATH = 999

DATA = DATASETS + DATAFILE
TRAININGDATA = DATASETS + "train.csv"
TESTINGDATA = DATASETS + "valid.csv"

#####prediction feature

TARGER = "Adj Close"


### Features
FEATURESSELECTED = ["Open","High","Low","Close","Volume"]

DROPFEATURES = ["Date"]

PREDICTIONWINDOW = 7