#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:18:01 2020

@author: llcnt
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

def sho(epochs, insert, embed, num, weight):
    with open("epochs_Acc1"+"_"+str(epochs)+"_"+str(insert)+"_"+str(embed)+"_"+str(num)+"_"+str(weight)+".txt", "rb") as fp:   #Pickling
        epochs_Acc1 = pickle.load(fp)
    plt.plot(epochs_Acc1)
    
def sho_per(epochs, insert, embed, num, weight):
    with open("epochs_train_res_perplexity"+"_"+str(epochs)+"_"+str(insert)+"_"+str(embed)+"_"+str(num)+"_"+str(weight)+".txt", "rb") as fp:   #Pickling
        epochs_Acc1 = pickle.load(fp)
    plt.plot(epochs_Acc1)
    
def sho_entropy(epochs, insert, embed, num, weight):
    with open("epochs_train_res_perplexity"+"_"+str(epochs)+"_"+str(insert)+"_"+str(embed)+"_"+str(num)+"_"+str(weight)+".txt", "rb") as fp:   #Pickling
        epochs_Acc1 = pickle.load(fp)
    plt.plot(np.log(epochs_Acc1))