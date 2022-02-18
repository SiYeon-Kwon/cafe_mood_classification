# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:03:05 2022

@author: USER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import cv2

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = 'C:/Windows/System32/kwontest/experiment'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

validation_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize((256,256)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transform)
test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 32)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32)
#print(type(train_loader))
# Label mapping

import json
with open('C:/Windows/System32/kwontest/experiment/train/mergefile.json', 'r') as f:
    json_file = json.load(f)
    
# Bulid and train the classifier
model = models.efficientnet_b7(pretrained=False)

# Freeze pretrained model parameters to avoid backpropagation
for parameter in model.parameters():
    parameter.requires_grad = False
    
from collections import OrderedDict

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2560, 32)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.3)),
                                        ('fc2', nn.Linear(32, 2560)),
                                        ('output', nn.LogSoftmax(dim=1))]))
model.classifier = classifier

#Function for the validation pass
def validation(model, validation_loader, criterion):
    
    val_loss = 0
    accuracy = 0
    
    for images, labels in iter(validation_loader):
        
        images, labels = images.to('cuda'), labels.to('cuda')
        
        output = model.forward(images)
        val_loss += criterion(output, labels).item()
        
        probabilities = torch.exp(output)
        
        equality = (labels.data == probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return val_loss, accuracy

# Loss functional and fradient descent

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# Train the classifier

from workspace_utils import active_session

def train_classifier():
    
    #with active_session():
        
        epochs = 100
        steps = 1000
        print_every = 10
        
        model.to('cuda')
        
        for e in range(epochs):
            
            model.train()
            
            running_loss = 0
            
            for images, labels in iter(train_loader):
                
                steps += 1
                
                images, labels = images.to('cuda'), labels.to('cuda')
                
                optimizer.zero_grad()
                
                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    
                    model.eval()
                    
                    #Turn off gradients for validation, save memory and computations
                    with torch.no_grad():
                        validation_loss, accuracy = validation(model, validation_loader, criterion)
                        
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_loader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
                    
                    running_loss = 0
                    model.train()
                    
train_classifier()



#Test Network

def test_accuracy(model, test_loader):
    
    # Do  validation on the test set
    model.eval()
    model.to('cuda')
    
    with torch.no_grad():
        
        accuracy = 0
        
        for images, labels in iter(test_loader):
            
            images, labels = images.to('cuda'), labels.to('cuda')
            
            output = model.forward(images)
            
            probabilities = torch.exp(output)
            
            equality = (labels.data == probabilities.max(dim=1)[1])
            
            accuracy += equality.type(torch.FloatTensor).mean()
            
        print("Test Accuracy: {}".format(accuracy/len(test_loader)))
        
test_accuracy(model, test_loader)
        


# Save the checkpoint

'''def save_checkpoint(model):
    
    model.class_to_idx = train_dataset.class_do_idx
    
    checkpoint = {'arch': "vgg16",
                   'class_to_idx': model.class_to_idx,
                   'model_state_dict': model.state_dict()
                   }
    
    torch.save(checkpoint, 'checkpoint.pth')
    
save_checkpoint(model)


#Loading the checkpoint

from collections import OrderedDict

# Function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'vgg16':
        
        model = models.vgg16(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
            
    else:
        print("Architecture no recognized.")
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(32, 2560)),
                                            ('relu', nn.ReLU()),
                                            ('drop', nn.Dropout(p=0.5)),
                                            ('fc2', nn.Linear(2560, 32)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model
'''

'''
#Image Preprocessing
from PIL import Images

def preprocess_images(image_path):
    
    pil_images = Image.open(image_path)
    
    #Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    #Crop
'''