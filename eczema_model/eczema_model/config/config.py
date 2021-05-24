# Import libraries 
from pathlib import Path

# Get main folder path
ROOT = f"{Path(__file__).resolve().parents[2]}"
ROOT = ROOT.split("/")[-1]

DATAPATH = ROOT + "\data"
TRAIN = ROOT + "/data/train"
VALID = ROOT + "/data/val"

BATCHSIZE = 20
NUMWORKERS = 0
NUMEPOCHS = 55
# print(VALID)
