#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:26:49 2019

@author: f
"""
import torch.nn as nn
import torch.nn.functional as F

##################################
# Define Networks
##################################

# =============================== Q(z|X) ======================================
class Q_net(nn.Module):
  ''' front end part of discriminator and Q'''

  def __init__(self):
    super(Q_net, self).__init__()

    self.main = nn.Sequential(
      nn.Conv2d(1, 64, 4, 2, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(64, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 1024, 7, bias=False),
      nn.BatchNorm2d(1024),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 128, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.1, inplace=True)
    )
    self.lin1 = nn.Linear(128,10)
    self.lin3 = nn.Linear(128,4)
    

  def forward(self, x):
    output = self.main(x)
    y = F.softmax(self.lin1(output.squeeze()))
    z_gauss = self.lin3(output.squeeze())
    return z_gauss,y

# =============================== P(X|z) ======================================
class P_net(nn.Module):

  def __init__(self):
    super(P_net, self).__init__()

    self.main = nn.Sequential(
      nn.ConvTranspose2d(14, 1024, 1, 1, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    output = self.main(x)
    return output

class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(4, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return self.lin3(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)