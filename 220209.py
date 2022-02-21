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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size = 16)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 16)
#print(type(train_loader))
# Label mapping

import json
with open('C:/Windows/System32/kwontest/experiment/train/mergefile.json', 'r') as f:
    json_file = json.load(f)
    
# Bulid and train the classifier
model = models.efficientnet_b7(pretrained=True)

# Freeze pretrained model parameters to avoid backpropagation
for parameter in model.parameters():
    parameter.requires_grad = False
    
from collections import OrderedDict

# Build custom classifier
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2560, 32)),
                                        ('relu', nn.ReLU()),
                                        ('drop', nn.Dropout(p=0.5)),
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

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Save model if validation loss decrease'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        

# Train the classifier

#from workspace_utils import active_session

def train_classifier():
    
    #with active_session():
        
        epochs = 100
        steps = 1000
        print_every = 10
        patience = 20
        
        model.to('cuda')
        
        early_stopping = EarlyStopping(patience = patience, verbose = True)
        
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
                    
                    early_stopping(validation_loss, model)
                    
                    if early_stopping.early_stop:
                     print("Early stopping")
                     break
                    
                    running_loss = 0
                    model.train()
                    
train_classifier()


# Save the checkpoint

def save_checkpoint(model):
    
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'arch': "efficientnet",
                   'class_to_idx': model.class_to_idx,
                   'model_state_dict': model.state_dict()
                   }
    
    torch.save(checkpoint, 'C:/Windows/System32/kwontest/checkpoint.pth')
    
save_checkpoint(model)


#Loading the checkpoint

from collections import OrderedDict

# Function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
    
    checkpoint = torch.load('C:/Windows/System32/kwontest/checkpoint.pth') #Or checkpoint.pt
    
    if checkpoint['arch'] == 'efficientnet':
        
        model = models.efficientnet_b7(pretrained=True)
        
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


#Image Preprocessing
from PIL import Image

def process_image(image_path):
    
    # Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image_path)
    
    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    # Crop 
    left_margin = (pil_image.width-224)/2
    bottom_margin = (pil_image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    if title is not None:
        ax.set_title(title)
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

image = process_image('C:/Windows/System32/kwontest/experiment/test/retro/91.jpg')
imshow(image)

# Implement the code to predict the class from an image file

def predict(image_path, model, topk=4): #Set the topk according to the number of image classes
    '''Predict the class (or classes) of an image using a trained deep learning model.
    '''

    image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    #print(image.shape)
    #print(type(image))
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    
    probabilities = torch.exp(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    # Convert topk_indices to the actual class labels using class_to_idx
    # Invert the dictionary so you get a mapping from index to class.
    
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    #print(idx_to_class)
    
    top_classes = [idx_to_class[index] for index in top_indices]
    
    return top_probabilities, top_classes
    
probs, classes = predict('C:/Windows/System32/kwontest/experiment/test/retro/91.jpg', model)   
print(probs)
print(classes)
