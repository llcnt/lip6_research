#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: llcnt
"""

import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.models as models

import torchvision.transforms as transforms

#import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("epochs", default= 1, type=int, help="nb of epochs to train on")
parser.add_argument("weight", default= 1, type=int, help="weight for the classification loss")
parser.add_argument("learning_rate", default= 1, type=float, help="learning rate")


args = parser.parse_args()


#%%
### Device and datasets
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
                                  ]))


validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
                                  ]))



#%%
### Stacked model (vgg+compression)

class myModel(nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        vgg_model = models.vgg16(pretrained=False)		
        vgg_list = list(vgg_model.features.children())
        self.final_list = []
        for i, layer in enumerate(vgg_list):
            self.final_list.append(layer)
            
        self.features = nn.Sequential(*self.final_list)
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
 	    
    def forward(self,x):
        for i, layer in enumerate(self.features):
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    
#%%
### Training
batch_size = 128


training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


def main():
    
    model = myModel().to(device)

    optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum = 0.9, weight_decay =  5e-4)

    
    epochs_train_res_classif_loss = []

    epochs_Acc1 = []
    epochs_Acc5 = []
    
    for epoch in range(args.epochs):
        
        ### adapt lr
        adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        
        ### Switch to train mode
        model.train()

        train_res_classif_loss = []

        print('%d epoch' % (epoch+1))
        for i, (images, target) in enumerate(training_loader): #50000/256 ~ 200 steps 
            images = images.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
        
            output= model(images)
            
            # print(output.shape, target.shape)
            classif_loss = nn.CrossEntropyLoss()(output, target)
            loss = args.weight*classif_loss
            
            loss.backward()
        
            optimizer.step()
            

            train_res_classif_loss.append(classif_loss.item())

        
            
        print('%d epoch' % (epoch+1))

        print('classif_loss: %.3f' % np.mean(train_res_classif_loss))

        print()

        epochs_train_res_classif_loss.append(np.mean(train_res_classif_loss))

        ### Evaluate on validation set
        model.eval()
        Acc1 = []
        Acc5 = []
        with torch.no_grad():
            for i, (images, target) in enumerate(validation_loader): #10000/256 ~ 40 steps
                images = images.to(device)
                target = target.to(device)
                # compute output
                output = model(images)
    
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                Acc1.append(acc1.cpu().numpy())
                Acc5.append(acc5.cpu().numpy())
                

        print('accuracy_top_1: %.3f' % np.mean(Acc1))
        print('accuracy_top_5: %.3f' % np.mean(Acc5))
        epochs_Acc1.append(np.mean(Acc1))
        epochs_Acc5.append(np.mean(Acc5))

#%%
### Adapt the learning rate through iterations
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 50 epochs"""
    lr = args.learning_rate * (0.5 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#%%
### Compute accuracy
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
#%%
if __name__ == '__main__':
    main()