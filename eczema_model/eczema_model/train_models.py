
from __future__ import print_function, division
from PIL.Image import NONE


import pandas as pd
import model as ml
from eczema_model import pipeline as pipeline
from eczema_model.config import config


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def train_evaluate():
    dataloaders, dataset_sizes, class_names = ml.load_datsets()
    model, criterion, optimizer, scheduler  = ml.loss_function_optimzer()
    model_ft = ml.train_model(model, criterion, optimizer, scheduler, dataset_sizes, dataloaders)
    
    return NONE

if __name__ == "__main__":
    print("Processing Train_models")
    df = train_evaluate()
    # print("Data splitting completed: ", df.head())