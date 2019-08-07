#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:10:20 2017

@author: pf ge
"""

import torch
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
import warnings
warnings.filterwarnings("ignore")


##################################
# Load test data
##################################
test_data = datasets.MNIST(root='./data', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor).cuda()/255.
test_y = test_data.test_labels

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

########functions##############
    
def cluster_acc(Y_pred, Y):
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in range(Y_pred.size):
      w[Y_pred[i], Y[i]] += 1
      ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w     
    


[Q, P] = torch.load('./model/mnist_test.pkl')


#test model
real_y = test_y.numpy()
Q.eval()        
_,y = Q(test_x)
pred_y = torch.max(y, 1)[1].cpu().data.squeeze()
label_y = pred_y.cpu().numpy()
acc = cluster_acc(label_y,real_y)

print(acc)

# fixed random variables
Idx = np.arange(10).repeat(10)
one_hot = np.zeros((100, 10))
one_hot[range(100), Idx] = 1
  
c = np.linspace(-2, 2, 10).reshape(1,-1)
c = np.repeat(c, 10 ,0).reshape(-1, 1)
c1 = np.hstack([c, np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)])
c2 = np.hstack([np.zeros_like(c), c, np.zeros_like(c), np.zeros_like(c)])
c3 = np.hstack([ np.zeros_like(c), np.zeros_like(c), c, np.zeros_like(c)])
c4 = np.hstack([np.zeros_like(c), np.zeros_like(c), np.zeros_like(c), c])
dis_c = torch.FloatTensor(100, 10).cuda()
con_c = torch.FloatTensor(100, 4).cuda()
dis_c = Variable(dis_c)
con_c = Variable(con_c)

#plot images
P.eval()
dis_c.data.copy_(torch.Tensor(one_hot))

con_c.data.copy_(torch.from_numpy(c1))
z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
x_save = P(z)
save_image(x_save.data, './tmp/gen_c1.png', nrow=10)

con_c.data.copy_(torch.from_numpy(c2))
z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
x_save = P(z)
save_image(x_save.data, './tmp/gen_c2.png', nrow=10)

con_c.data.copy_(torch.from_numpy(c3))
z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
x_save = P(z)
save_image(x_save.data, './tmp/gen_c3.png', nrow=10)

con_c.data.copy_(torch.from_numpy(c4))
z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
x_save = P(z)
save_image(x_save.data, './tmp/gen_c4.png', nrow=10)

    