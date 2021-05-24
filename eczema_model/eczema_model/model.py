from __future__ import print_function, division

from PIL.Image import NONE
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim

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
    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'eczema_model/data/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print("class_names", class_names)

    return dataloaders, dataset_sizes, class_names


# Select pretrained model
def load_pretrained_model(display_arch=False):
    mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    if(display_arch):
        print(mobilenet_v2)

    return mobilenet_v2

def freezing_pretrained_model(model, class_names):
    for param in model.features.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[1].in_features

    last_layer = nn.Linear(n_inputs, len(class_names))

    model.classifier[1] = last_layer

    # if GPU is available, move the model to GPU
    if Is_gpu_avaliable():
        model.cuda()

    # check to see that your last layer produces the expected number of outputs
    print("The number of ouput classes are: ", model.classifier[1].out_features)

    return model 


# Define pytorch model
def pretrained_model_tuning():
    #classes are folders in each directory with these names
    classes = ['Atopic dermatitis', 'Neurodermatitis', 'Stasis dermatitis','Contact dermatitis',
      'Nummular eczema','Dyshidrotic eczema', 'Seborrheic dermatitis']


    dataloaders, dataset_sizes, class_names = load_datsets()

    # Load pretained model
    mobile_net = load_pretrained_model(display_arch=True)


    #freeze the last layer and modify the last layer
    mobile_net = freezing_pretrained_model(mobile_net, class_names)

    return mobile_net


def train_model(model, criterion, optimizer, scheduler, dataset_size, dataloaders):
    num_epochs = config.NUMEPOCHS
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(Is_gpu_avaliable())
                labels = labels.to(Is_gpu_avaliable())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects.double() / dataset_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def loss_function_optimzer():

    model_ft = pretrained_model_tuning()

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler 


