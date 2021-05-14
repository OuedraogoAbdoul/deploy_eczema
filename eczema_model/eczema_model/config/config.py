# Import libraries 
from pathlib import Path

# Get main folder path
ROOT = f"{Path(__file__).resolve().parents[2]}"
ROOT = ROOT.split("/")[-1]

TRAIN = ROOT + "/data/train"
VALID = ROOT + "/data/val"
# print(VALID)
