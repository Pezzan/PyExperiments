#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import torch
from torch import nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as dset
from IPython.display import display, clear_output
from torch.autograd import Variable


# In[3]:


def status(batch_size, ep, epoch, i, loss, data_loader):
    # status
    clear_output(wait=True)
    print(str(ep) + '/' + str(epoch))
    print('batch: ' + str(i+1) + '/' + str(len(data_loader)) + 
             ' [' + '='*int((i+1)/(len(data_loader)/20)) +
              '>' + ' '*(20 - int((i+1)/(len(data_loader)/20))) +
              ']')
    
    #print('Loss: %.4g '% ((loss / ((i+1)*batch_size))))
    print('Loss: %.4g '% (loss))
    
#-------------------------------------------------------------------
def showAllImages(x,y,z):
    x = x[1,:,:,:].detach()
    y = y[1,:,:,:].detach()
    z = z[1,:,:,:].detach()
    
    x = x.cpu()
    y = y.cpu()
    z = z.cpu()
    
    plt.figure(figsize=(12,8))
    plt.subplot(131)
    plt.imshow(np.transpose(x, (1,2,0)))
    plt.subplot(132)
    plt.imshow(np.transpose(y, (1,2,0)))
    plt.subplot(133)
    plt.imshow(np.transpose(z, (1,2,0)))
    plt.show()

#-------------------------------------------------------------------
def conv(dimIn, dimOut):
    model = nn.Sequential(
        nn.Conv2d(dimIn, dimOut, kernel_size=3, stride=1,
                  padding=1),
        nn.BatchNorm2d(dimOut),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(dimOut, dimOut, kernel_size=3, stride=1,
                 padding=1),
        nn.BatchNorm2d(dimOut)
    )
    return model

#-------------------------------------------------------------------
def pool():
    p = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return p

#-------------------------------------------------------------------
def invConv(dimIn, dimOut):
    model = nn.Sequential(
        nn.ConvTranspose2d(dimIn, dimOut, kernel_size=3, stride=2,
                           padding=1,output_padding=1),
        nn.BatchNorm2d(dimOut),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return model
    
#-------------------------------------------------------------------
def last(dimIn, dimOut):
    model = nn.Sequential(
        nn.Conv2d(dimIn, dimOut, kernel_size=3, stride=1,
                  padding=1),
        nn.Tanh()
    )
    return model


# In[4]:


class TestGen(nn.Module):
    def __init__(self, nFilters):
        super().__init__()
        self.fil = nFilters
    
        self.conv1 = conv(3, self.fil)
        self.pool1 = pool()
        
        self.bridge = conv(self.fil, self.fil*2)
        
        self.inv1 = invConv(self.fil*2, self.fil)
        self.up1 = conv(self.fil*2, self.fil)
        
        self.last = last(self.fil, 3)
        
    def forward(self, img):
        conv1 = self.conv1(img)
        pool1 = self.pool1(conv1)
        
        bridge = self.bridge(pool1)
        
        inv1 = self.inv1(bridge)
        join1 = torch.cat([inv1, conv1],dim=1)
        up1 = self.up1(join1)
        
        res = self.last(up1)
        return res
        


# In[ ]:




