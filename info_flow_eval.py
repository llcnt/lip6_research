# -*- coding: utf-8 -*-
"""
@author: llcnt
"""

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.models as models

import torchvision.transforms as transforms

import pickle

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("epochs", default= 1, type=int, help="nb of epochs to train on")
parser.add_argument("insertion_place", default= 1, type=int, help="int where to insert the compression module")
parser.add_argument("embedding_dim", default= 64, type=int, help="dimension of the latent vae embedding")
parser.add_argument("num_embeddings", default= 512, type=int, help="nb of embeddings in the codebook")
parser.add_argument("weight", default= 1, type=int, help="weight for the classification loss")

args = parser.parse_args()

# class Arguments():
#     def __init__(self):
#         self.epochs = 1#00
#         self.insertion_place = 1
#         self.embedding_dim = 64
#         self. num_embeddings = 512
        
# args = Arguments()

#%%
### Device and datasets
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
                                  ]))

data_variance = np.var(training_data.data / 255.0)

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
                                  ]))



# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  ### For ImageNet
#                                  std=[0.229, 0.224, 0.225])

#%%
### Compression module

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        return self._residual_stack(x)
    
class Decoder(nn.Module):
    def __init__(self, out_channels, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, 
                                                stride=1, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=out_channels,
                                                kernel_size=3, 
                                                stride=1, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        return self._conv_trans_2(x)
    
#%%
### Compression model

class compression_model(nn.Module):
    def __init__(self, input_encoder_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(compression_model, self).__init__()
        
        self._encoder = Encoder(input_encoder_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)

        self._decoder = Decoder(input_encoder_channels, embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        
        return loss, x_recon, perplexity

#%%
### Stacked model (vgg+compression)

class myModel(nn.Module):
    def __init__(self, insertion_layer, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay):
        super(myModel,self).__init__()
        vgg_model = models.vgg16(pretrained=False)		
        vgg_list = list(vgg_model.features.children())
        self.final_list = []
        for i, layer in enumerate(vgg_list):
            if i == insertion_layer:
                input_encoder_channels = vgg_list[i-1].out_channels 
                vqvae = compression_model(input_encoder_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay)
                self.final_list.append(vqvae)
                self.is_compression = len(self.final_list) - 1
            self.final_list.append(layer)
        self.features = nn.Sequential(*self.final_list)
        self.codebook = vqvae._vq_vae._embedding.weight
        
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
            if i == self.is_compression:
                data_before = x
                loss, x, perplexity = layer(x)
                data_recon = x
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        return loss, self.classifier(x), perplexity, data_before, data_recon
    
    
#%%
### Training
batch_size = 128
# num_training_updates = 15000

num_hiddens = 256
num_residual_hiddens = 256
num_residual_layers = 2

# embedding_dim = 64
# num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 2e-4

training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

model = myModel(args.insertion_place, num_hiddens, num_residual_layers, num_residual_hiddens,
              args.num_embeddings, args.embedding_dim, 
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), learning_rate)

def main():
    
    epochs_train_res_recon_error = []
    epochs_train_res_perplexity = []
    epochs_train_res_classif_loss = []
    epochs_train_res_vq_loss = []

    epochs_Acc1 = []
    epochs_Acc5 = []
    
    for epoch in range(args.epochs):
        ### Switch to train mode
        model.train()
        train_res_recon_error = []
        train_res_perplexity = []
        train_res_classif_loss = []
        train_res_vq_loss = []

        print('%d epoch' % (epoch+1))
        for i, (images, target) in enumerate(training_loader): #50000/256 ~ 200 steps 
            images = images.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
        
            vq_loss, output, perplexity, data_before, data_recon = model(images)

            recon_error = F.mse_loss(data_recon, data_before) / data_variance
            loss = recon_error + vq_loss
            
            # print(output.shape, target.shape)
            classif_loss = nn.CrossEntropyLoss()(output, target)
            loss += args.weight*classif_loss
            
            loss.backward()
        
            optimizer.step()
            
            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())
            train_res_classif_loss.append(classif_loss.item())
            train_res_vq_loss.append(vq_loss.item())

        
            
        print('%d epoch' % (epoch+1))
        print('recon_error: %.3f' % np.mean(train_res_recon_error))
        print('perplexity: %.3f' % np.mean(train_res_perplexity))
        print('classif_loss: %.3f' % np.mean(train_res_classif_loss))
        print('vq_loss: %.3f' % np.mean(train_res_vq_loss))

        print()
        epochs_train_res_recon_error.append(np.mean(train_res_recon_error))
        epochs_train_res_perplexity.append(np.mean(train_res_perplexity))
        epochs_train_res_classif_loss.append(np.mean(train_res_classif_loss))
        epochs_train_res_vq_loss.append(np.mean(train_res_vq_loss))

        ### Evaluate on validation set
        model.eval()
        Acc1 = []
        Acc5 = []
        with torch.no_grad():
            for i, (images, target) in enumerate(validation_loader): #10000/256 ~ 40 steps
                images = images.to(device)
                target = target.to(device)
                # compute output
                vq_loss, output, perplexity, data_before, data_recon = model(images)
    
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                Acc1.append(acc1.cpu().numpy())
                Acc5.append(acc5.cpu().numpy())
                

        print('accuracy_top_1: %.3f' % np.mean(Acc1))
        print('accuracy_top_5: %.3f' % np.mean(Acc5))
        epochs_Acc1.append(np.mean(Acc1))
        epochs_Acc5.append(np.mean(Acc5))


    with open("epochs_train_res_recon_error"+"_"+str(args.epochs)+"_"+str(args.insertion_place)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.weight)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_train_res_recon_error, fp)
    with open("epochs_train_res_perplexity"+"_"+str(args.epochs)+"_"+str(args.insertion_place)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.weight)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_train_res_perplexity, fp)
    with open("epochs_train_res_classif_loss"+"_"+str(args.epochs)+"_"+str(args.insertion_place)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.weight)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_train_res_classif_loss, fp)
    with open("epochs_train_res_vq_loss"+"_"+str(args.epochs)+"_"+str(args.insertion_place)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.weight)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_train_res_vq_loss, fp)
    with open("epochs_Acc1"+"_"+str(args.epochs)+"_"+str(args.insertion_place)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.weight)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_Acc1, fp)
    with open("epochs_Acc5"+"_"+str(args.epochs)+"_"+str(args.insertion_place)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.weight)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_Acc5, fp)

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