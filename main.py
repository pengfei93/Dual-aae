#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:15:59 2019

@author: f
"""

from model import *
from trainer import Trainer
import torch


Q = Q_net()
P = P_net()
D_gauss = D_net_gauss()


for i in [Q, P, D_gauss]:
  i.cuda()
  i.apply(weights_init)

#pre-train model by AE
[Q,P,D_gauss]= torch.load('./model/mnist_pretrain.pkl')  
  
trainer = Trainer(Q, P, D_gauss)
trainer.train()