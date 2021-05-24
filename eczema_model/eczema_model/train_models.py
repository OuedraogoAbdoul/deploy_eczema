
from __future__ import print_function, division


import pandas as pd
import model as ml
from eczema_model import pipeline as pipeline


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


def train_models():
    #classes are folders in each directory with these names
    classes = ['Atopic dermatitis', 'Neurodermatitis', 'Stasis dermatitis','Contact dermatitis',
      'Nummular eczema','Dyshidrotic eczema', 'Seborrheic dermatitis']


    train_loader = ml.load_datsets()

    # Visualize some sample data
    ml.visualize_sample_images(train_loader, classes)


    # Load pretained model
    mobile_net = ml.load_pretrained_model(display_arch=True)



    return train_loader

if __name__ == "__main__":
    print("Processing Train_models")
    df = train_models()
    # print("Data splitting completed: ", df.head())