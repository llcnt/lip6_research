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
parser.add_argument("embedding_dim", default= 64, type=int, help="dimension of the latent vae embedding")
parser.add_argument("num_embeddings", default= 512, type=int, help="nb of embeddings in the codebook")
parser.add_argument("gpu", default= 1, type=int, help="which gpu to use")
parser.add_argument("learning_rate", default= 5e-2, type=float, help="learning rate")

args = parser.parse_args()



#%%
### Device and datasets
device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="/data/leconte/data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
                                  ]))

data_variance = np.var(training_data.data / 255.0)

validation_data = datasets.CIFAR10(root="/data/leconte/data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.49139968,  0.48215841,  0.44653091), (0.24703223,  0.24348513,  0.26158784))
                                  ]))

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
        
        self.update_codebook = True

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
        if self.training and self.update_codebook:
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
            # print('codebook', self._embedding.weight)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    
    
#%%
### Compression model
# def whiten(x):
#     b, c = x.size(0), x.size(1)
#     x = x - torch.mean(x, dim=0, keepdim=True)
#     x = x.view(b, c, -1)
#     sigma = torch.matmul(x.permute(2, 1, 0).contiguous(), x.permute(2, 0, 1).contiguous()) / (c-1)   #row means were estimated from the data.
#     print(sigma.size())
#     u, s, v = torch.svd(sigma.cpu())
#     print(u.size())
#     sigma_inv_rac = torch.matmul(torch.matmul(u, torch.diag_embed(1/torch.sqrt(s+1e-5))), u.transpose(1, 2))
    
#     return  sigma, sigma_inv_rac

class compression_model(nn.Module):
    def __init__(self, input_encoder_channels, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(compression_model, self).__init__()
        

        self._pre_vq_conv = nn.Conv2d(in_channels=input_encoder_channels, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        
        self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)

        self._post_vq_conv = nn.Conv2d(in_channels=embedding_dim, 
                                      out_channels=input_encoder_channels,
                                      kernel_size=1, 
                                      stride=1)

    def forward(self, x):
        # b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        # sigma, sigma_inv_rac = whiten(x)
        # mu = torch.mean(x, dim=0, keepdim=True)
        # x = x - mu
        # x = x.view(b, c, -1)
        # x = torch.matmul(sigma_inv_rac, x.permute(2, 1, 0).contiguous())
        # x = x.permute(2, 1, 0).contiguous().view(b, c, h, w)
        
        x = self._pre_vq_conv(x)
        loss, quantized, perplexity, encodings = self._vq_vae(x)
        codebook = self._vq_vae._embedding.weight
        x_recon = self._post_vq_conv(quantized)
                
        # x_recon = x_recon.view(b, c, -1)
        # x_recon = torch.matmul(sigma, torch.matmul(sigma_inv_rac, x_recon.permute(2, 1, 0).contiguous()))
        # x_recon = x_recon.permute(2, 1, 0).contiguous().view(b, c, h, w)
        # x_recon += mu
        
        
        return loss, x_recon, perplexity, codebook

#%%
### vgg model

class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel,self).__init__()
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
### Stacked model (vgg+compression)

class myModel(nn.Module):
    def __init__(self, insertion_layer, num_embeddings, embedding_dim, commitment_cost, decay):
        super(myModel,self).__init__()
        vgg_model = VGGModel()	
        # vgg_model.load_state_dict(torch.load('/data/leconte/vggmodel.pt', map_location="cuda:"+str(args.gpu)))
        vgg_list = list(vgg_model.features.children()) + list(vgg_model.classifier.children())
        self.final_list = []
        self.final_list_classif = []

        for i, layer in enumerate(list(vgg_model.features.children())):
            for param in layer.parameters():
                param.requires_grad = True
            if i == insertion_layer:
                input_encoder_channels = vgg_list[i-1].out_channels 
                vqvae = compression_model(input_encoder_channels, num_embeddings, embedding_dim, commitment_cost, decay)
                self.final_list.append(vqvae)
                self.is_compression = len(self.final_list) - 1
            self.final_list.append(layer)
        self.features = nn.Sequential(*self.final_list)
        self.codebook = self.features[self.is_compression]._vq_vae._embedding.weight
        
        for i, layer in enumerate(list(vgg_model.classifier.children())):
            for param in layer.parameters():
                param.requires_grad = True
            self.final_list_classif.append(layer)
        self.classifier = nn.Sequential(*self.final_list_classif)
        
 	    
    def forward(self,x, mu, sigma, sigma_inv_rac):
        for i, layer in enumerate(self.features):
            if i == self.is_compression:
                data_before = x
                
                b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
                x = x - mu.view(1, c, 1, 1)
                x = x.view(b, c, -1)
                x = torch.matmul(sigma_inv_rac, x.permute(2, 1, 0).contiguous())
                x = x.permute(2, 1, 0).contiguous().view(b, c, h, w)
                
                loss, x, perplexity, codebook = layer(x)
                
                x = x.view(b, c, -1)
                x = torch.matmul(sigma, torch.matmul(sigma_inv_rac, x.permute(2, 1, 0).contiguous()))
                x = x.permute(2, 1, 0).contiguous().view(b, c, h, w)
                x += mu.view(1, c, 1, 1)
                
                data_recon = x
            else:
                x = layer(x)
        x = x.view(x.size(0), -1)
        return loss, self.classifier(x), perplexity, data_before, data_recon, codebook
    
# #%%
# ###  model (vgg+)

# class model_vgg_cut(nn.Module):
#     def __init__(self, insertion_layer):
#         super(model_vgg_cut,self).__init__()
#         vgg_model = VGGModel()	
#         vgg_model.load_state_dict(torch.load('/data/leconte/vggmodel.pt', map_location="cuda:"+str(args.gpu)))
#         vgg_list = list(vgg_model.features.children())[:insertion_layer]
#         self.final_list = []

#         for i, layer in enumerate(vgg_list):
#             for param in layer.parameters():
#                 param.requires_grad = False
#             self.final_list.append(layer)
#         self.features = nn.Sequential(*self.final_list)
        
 	    
#     def forward(self,x):
#         for i, layer in enumerate(self.features):
#             x = layer(x)
#         return x
#%%
### Training
batch_size = 128 # 32 #64 #128

num_hiddens = 256
num_residual_hiddens = 256
num_residual_layers = 2

commitment_cost = 0.25

decay = 0.99


training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


def insertion_and_train(ins):
    
    model = myModel(ins,
              args.num_embeddings, args.embedding_dim, 
              commitment_cost, decay).to(device)

    # optimizer = optim.SGD(model.parameters(), args.learning_rate, momentum = 0.9, weight_decay =  5e-4)
    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    
    epochs_train_res_recon_error = []
    epochs_train_res_perplexity = []
    epochs_train_res_classif_loss = []
    epochs_train_res_vq_loss = []

    epochs_Acc1 = []
    epochs_Acc5 = []
  
    train_epochs_Acc1 = []
    train_epochs_Acc5 = []
    
  
    ###
    coherence_matrices = []
    print(torch.matmul(model.codebook/torch.norm(model.codebook, dim=1).view((model.codebook.size(0),1)), (model.codebook/torch.norm(model.codebook, dim=1).view((model.codebook.size(0),1))).t()))
    coherence_matrices.append(torch.matmul(model.codebook/torch.norm(model.codebook, dim=1).view((model.codebook.size(0),1)), (model.codebook/torch.norm(model.codebook, dim=1).view((model.codebook.size(0),1))).t()).cpu().detach())

    for epoch in range(args.epochs):

        with torch.no_grad():  
            ### Whitening on training set
            for i, (images, target) in enumerate(DataLoader(training_data, 
                                     batch_size=256, 
                                     shuffle=True,
                                     pin_memory=True)): #50000/256 ~ 200 steps 
                images = images.to(device)
                _, _, _, x, _, _ = model(images, torch.zeros(args.embedding_dim).to(device), torch.zeros((args.embedding_dim, args.embedding_dim)).to(device), torch.zeros((args.embedding_dim, args.embedding_dim)).to(device))

                b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
                if i == 0 :
                    mu = torch.zeros(c).to(device)
                mu += (b/50000)*torch.mean(x, dim=[0,2,3], keepdim=False)
                
            for i, (images, target) in enumerate(DataLoader(training_data, 
                                     batch_size=256, 
                                     shuffle=True,
                                     pin_memory=True)): #50000/256 ~ 200 steps 
                images = images.to(device)
                _, _, _, x, _, _ = model(images, torch.zeros(args.embedding_dim).to(device), torch.zeros((args.embedding_dim, args.embedding_dim)).to(device), torch.zeros((args.embedding_dim, args.embedding_dim)).to(device))
                b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
                if i == 0 :
                    sigma = torch.zeros((c, c)).to(device)
                x -= mu.view(1, c, 1, 1)
                x = x.view(b, c, -1)
                sigma += (1/49999)*torch.mean(torch.matmul(x.permute(2, 1, 0).contiguous(), x.permute(2, 0, 1).contiguous()), dim=0, keepdim=False) #/ (c-1)   #row means were estimated from the data.
                            
            u, s, v = torch.svd(sigma.cpu())
            sigma_inv_rac = torch.matmul(torch.matmul(u, torch.diag_embed(1/torch.sqrt(s+1e-1))), u.t())
        
        
        ### adapt lr
        adjust_learning_rate(optimizer, epoch)
        
        
        ### Switch to train mode
        model.train()
        
        train_res_recon_error = []
        train_res_perplexity = []
        train_res_classif_loss = []
        train_res_vq_loss = []
        
        print('%d epoch' % (epoch+1))
        for i, (images, target) in enumerate(training_loader): #50000/256 ~ 200 steps 
        
        ### the codebook is updated only 1/20 times
            if i%1==0:
                model.features[model.is_compression]._vq_vae.update_codebook = True
            else:
                model.features[model.is_compression]._vq_vae.update_codebook = False
                
            images = images.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
        
            vq_loss, output, perplexity, data_before, data_recon, codebook = model(images, mu.to(device), sigma.to(device), sigma_inv_rac.to(device))

            data_variance = torch.var(data_before)
            # print(data_variance.item())

            recon_error = F.mse_loss(data_recon, data_before) / data_variance
            loss = recon_error + vq_loss
            
            # print(output.shape, target.shape)
            classif_loss = nn.CrossEntropyLoss()(output, target)
            
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
        
        print(torch.matmul(codebook/torch.norm(codebook, dim=1).view((codebook.size(0),1)), (codebook/torch.norm(codebook, dim=1).view((codebook.size(0),1))).t()))
        coherence_matrices.append(torch.matmul(codebook/torch.norm(codebook, dim=1).view((codebook.size(0),1)), (codebook/torch.norm(codebook, dim=1).view((codebook.size(0),1))).t()).cpu().detach())


        epochs_train_res_recon_error.append(np.mean(train_res_recon_error))
        epochs_train_res_perplexity.append(np.mean(train_res_perplexity))
        epochs_train_res_classif_loss.append(np.mean(train_res_classif_loss))
        epochs_train_res_vq_loss.append(np.mean(train_res_vq_loss))
        
        ### Evaluate on train set
        model.eval()
        train_Acc1 = []
        train_Acc5 = []
        with torch.no_grad():
            for i, (images, target) in enumerate(training_loader): 
                images = images.to(device)
                target = target.to(device)
                # compute output
                vq_loss, output, perplexity, data_before, data_recon, codebook = model(images, mu.to(device), sigma.to(device), sigma_inv_rac.to(device))
    
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                train_Acc1.append(acc1.cpu().numpy())
                train_Acc5.append(acc5.cpu().numpy())
        print('train_accuracy_top_1: %.3f' % np.mean(train_Acc1))
        print('train_accuracy_top_5: %.3f' % np.mean(train_Acc5))
        train_epochs_Acc1.append(np.mean(train_Acc1))
        train_epochs_Acc5.append(np.mean(train_Acc5))
                
        ### Evaluate on validation set
        model.eval()
        Acc1 = []
        Acc5 = []
        with torch.no_grad():
            for i, (images, target) in enumerate(validation_loader): #10000/256 ~ 40 steps
                images = images.to(device)
                target = target.to(device)
                # compute output
                vq_loss, output, perplexity, data_before, data_recon, codebook = model(images, mu.to(device), sigma.to(device), sigma_inv_rac.to(device))
    
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                Acc1.append(acc1.cpu().numpy())
                Acc5.append(acc5.cpu().numpy())
                

        print('accuracy_top_1: %.3f' % np.mean(Acc1))
        print('accuracy_top_5: %.3f' % np.mean(Acc5))
        epochs_Acc1.append(np.mean(Acc1))
        epochs_Acc5.append(np.mean(Acc5))
        
    with open("/data/leconte/co_mat4/coherence_matrices"+"_"+str(ins)+'_'+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
        pickle.dump(coherence_matrices, fp)
        
    return epochs_train_res_recon_error, epochs_train_res_perplexity, epochs_train_res_classif_loss, epochs_train_res_vq_loss, epochs_Acc1, epochs_Acc5, train_epochs_Acc1, train_epochs_Acc5

#%%
def main():
    epochs_train_res_recon_error = []
    epochs_train_res_perplexity = []
    epochs_train_res_classif_loss = []
    epochs_train_res_vq_loss = []

    epochs_Acc1 = []
    epochs_Acc5 = []
  
    train_epochs_Acc1 = []
    train_epochs_Acc5 = []
    
    for ins in [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29]:
        print("Insertion in between :")
        print(ins)
        recon_error, perplexity, classif_loss, vq_loss, Acc1, Acc5, train_Acc1, train_Acc5 = insertion_and_train(ins)
        
        epochs_train_res_recon_error.append(recon_error)
        epochs_train_res_perplexity.append(perplexity)
        epochs_train_res_classif_loss.append(classif_loss)
        epochs_train_res_vq_loss.append(vq_loss)
        epochs_Acc1.append(Acc1)
        epochs_Acc5.append(Acc5)
        train_epochs_Acc1.append(train_Acc1)
        train_epochs_Acc5.append(train_Acc5)
        
        
        
        
    # with open("pretrained_train_res_recon_error"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
    #     pickle.dump(epochs_train_res_recon_error, fp)
    # with open("pretrained_train_res_perplexity"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
    #     pickle.dump(epochs_train_res_perplexity, fp)
    # with open("pretrained_train_res_classif_loss"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
    #     pickle.dump(epochs_train_res_classif_loss, fp)
    # with open("pretrained_train_res_vq_loss"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
    #     pickle.dump(epochs_train_res_vq_loss, fp)
    with open("/data/leconte/co_mat4/Acc1"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
        pickle.dump(epochs_Acc1, fp)
    # with open("pretrained_Acc5"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
    #     pickle.dump(epochs_Acc5, fp)
    with open("/data/leconte/co_mat4/train_Acc1"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
        pickle.dump(train_epochs_Acc1, fp)
    # with open("pretrained_train_Acc5"+"_"+str(args.epochs)+"_"+str(args.embedding_dim)+"_"+str(args.num_embeddings)+"_"+str(args.gpu)+"_"+str(args.learning_rate)+".txt", "wb") as fp:   #Pickling
    #     pickle.dump(train_epochs_Acc5, fp)
        
#%%
### Adapt the learning rate through iterations
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.learning_rate * (0.5 ** (epoch // 30))
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