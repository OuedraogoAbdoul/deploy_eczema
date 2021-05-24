from __future__ import print_function, division

from PIL.Image import NONE

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

plt.ion()   # interactive mode

from eczema_model.config import config
# print(config.DATAPATH)

# Check GPU presence
def Is_gpu_avaliable():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

# Load Data
def load_datsets():
    data_dir = 'eczema_model/data/'
    train_dir = os.path.join(data_dir, 'train/')
    test_dir = os.path.join(data_dir, 'val/')

    # load and transform data using ImageFolder

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                        transforms.ToTensor()])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)

    # print out some data stats
    print('Num training images: ', len(train_data))
    print('Num test images: ', len(test_data))


    # define dataloader parameters
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCHSIZE, 
                                            num_workers=config.NUMWORKERS, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config.BATCHSIZE, 
                                            num_workers=config.NUMWORKERS, shuffle=True)

    
    return train_loader

# Visualize a few images
def visualize_sample_images(inp, classes):
    # obtain one batch of training images
    dataiter = iter(inp)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        ax.set_title(classes[labels[idx]])
    plt.pause(10)


# Select pretrained model
def load_pretrained_model(display_arch=False):
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    if(display_arch):
        print(mobilenet_v2)

    return mobilenet_v2


# Define pytorch model
