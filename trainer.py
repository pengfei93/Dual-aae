#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:36:44 2019

@author: f
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class Trainer:

    def __init__(self, Q, P, D_gauss):

        self.q = Q
        self.p = P
        self.d = D_gauss
        
        self.batch_size = 100
        self.epoch = 300

    def sample_categorical(self, n_classes=10):
        '''
         Sample from a categorical distribution
         of size batch_size and # of classes n_classes
         return: torch.autograd.Variable with the sample
        '''
        idx = np.random.randint(0, 10, self.batch_size)
        cat = np.eye(n_classes)[idx].astype('float32')
        cat = torch.from_numpy(cat)
        return Variable(cat).cuda(),idx

    def zerograd(self):
        self.q.zero_grad()
        self.p.zero_grad()
        self.d.zero_grad()
            
    def d_entropy1(self, y):
        y1 = torch.mean(y,0)
        y2 = torch.sum(-y1*torch.log(y1+1e-5))
        return y2   
        
    def d_entropy2(self, y):
        y1 = -y*torch.log(y+1e-5)
        y2 = torch.sum(y1)/self.batch_size
        return y2      
        
    def cluster_acc(self, Y_pred, Y):
        assert Y_pred.size == Y.size
        D = max(Y_pred.max(), Y.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(Y_pred.size):
            w[Y_pred[i], Y[i]] += 1
            ind = linear_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w     

    def train(self):
        
        ##################################
        # Load data and create Data loaders
        ##################################
        # Mnist digits dataset
        train_data = datasets.MNIST(
            root = './data',
            train = True,                                     # this is training data
            transform = transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                            # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
            download = 1                        # download it if you don't have it
        )
        
        
        train_loader = DataLoader(dataset = train_data, batch_size = self.batch_size, shuffle = True, num_workers = 8)
        #test data
        test_data = datasets.MNIST(root='./data', train = False)
        test_x = Variable(torch.unsqueeze(test_data.test_data, dim = 1), volatile = True).type(torch.FloatTensor).cuda()/255.
        test_y = test_data.test_labels
        
        #loss function
        mse_dis = nn.MSELoss().cuda()
        criterionQ_dis = nn.NLLLoss().cuda()
        
        RE_solver = optim.Adam([{'params':self.q.parameters()}, {'params':self.p.parameters()}], lr=0.0001)
        I_solver = optim.Adam([{'params':self.q.parameters()}, {'params':self.p.parameters()}], lr=0.00001)
        Q_generator_solver = optim.Adam([{'params':self.q.parameters()}], lr=0.00005)
        D_loss_solver = optim.Adam([{'params':self.d.parameters()}], lr=0.00005)
        h_solver = optim.Adam([{'params':self.q.parameters()}], lr=0.00005)
        
        # fixed random variables, for plot image
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
        
        ACC = np.zeros(self.epoch)
        
        #train network
        for epoch in range(self.epoch):
            print('\n','Epoch:',epoch,'\n')
            
            # Set the networks in train mode (apply dropout when needed)
            self.q.train()
            self.p.train()
            self.d.train()
            # Loop through the labeled and unlabeled dataset getting one batch of samples from each
            # The batch size has to be a divisor of the size of the dataset or it will return
            # invalid samples
            
            for X, _ in tqdm(train_loader):
        
                X = Variable(X).cuda()
                
                # Init gradients
                self.zerograd()
        
                # D loss
                z_real_gauss = Variable(torch.randn(self.batch_size, 4))
                z_real_gauss = z_real_gauss.cuda()
            
                z_fake_gauss,_ = self.q(X)
            
                D_real_gauss = self.d(z_real_gauss)
                D_fake_gauss = self.d(z_fake_gauss.resize(self.batch_size, 4))
                D_loss =  -torch.mean(D_real_gauss ) + torch.mean(D_fake_gauss)
            
                D_loss.backward()
                D_loss_solver.step()
            
                # Init gradients
                self.zerograd()
                
                # G loss
                z_fake_gauss,_ = self.q(X)
                D_fake_gauss = self.d(z_fake_gauss.resize(self.batch_size, 4))
            
                G_loss = -torch.mean(D_fake_gauss)
                G_loss.backward()
                Q_generator_solver.step()
                
                # Init gradients
                self.zerograd()
        
                #H loss
                _, y = self.q(X)
                h1_loss = self.d_entropy2(y)
                h2_loss = -self.d_entropy1(y)
                h_loss = 0.5*h1_loss+h2_loss
                h_loss.backward()
                h_solver.step()
                
                # Init gradients
                self.zerograd()
                
                #recon_x
                z, y = self.q(X)
                X_re = self.p(torch.cat((z,y),1).resize(self.batch_size,14,1,1))
                
                recon_loss = mse_dis(X_re, X)
                recon_loss.backward()
                RE_solver.step()
                
                # Init gradients
                self.zerograd()
                
                #recon y and d
                y,idx = self.sample_categorical(n_classes=10)
                z = Variable(torch.randn(self.batch_size, 4)).cuda()
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                
                X_sample = self.p(torch.cat((z,y),1).resize(self.batch_size,14,1,1))
                z_recon, y_recon = self.q(X_sample)
                
                I_loss = criterionQ_dis(torch.log(y_recon), target) + mse_dis(z_recon[:,1:4], z[:,1:4])
                
                I_loss.backward()
                I_solver.step()
                
                # Init gradients
                self.zerograd()
                
            #test model  
            self.q.eval()  
            _,y = self.q(test_x)
            pred_y = torch.max(y, 1)[1].cpu().data.squeeze()
            label_y = pred_y.cpu().numpy()
            real_y = test_y.numpy()
            acc = self.cluster_acc(label_y,real_y)
            print('h_loss: {:.4}; D_gauss: {:.4}; G_gauss: {:.4}; recon_loss: {:.4}; I_loss: {:.4}; acc:'.format(h_loss.data[0], D_loss.data[0], G_loss.data[0], recon_loss.data[0], I_loss.data[0]), acc[0])
            ACC[epoch] = acc[0]
            
            #plot images
            self.p.eval()
            dis_c.data.copy_(torch.Tensor(one_hot))
            
            con_c.data.copy_(torch.from_numpy(c1))
            z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
            x_save = self.p(z)
            save_image(x_save.data, './tmp/g_%d_c1.png'%epoch, nrow=10)
    
            con_c.data.copy_(torch.from_numpy(c2))
            z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
            x_save = self.p(z)
            save_image(x_save.data, './tmp/g_%d_c2.png'%epoch, nrow=10)
    
            con_c.data.copy_(torch.from_numpy(c3))
            z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
            x_save = self.p(z)
            save_image(x_save.data, './tmp/g_%d_c3.png'%epoch, nrow=10)
    
            con_c.data.copy_(torch.from_numpy(c4))
            z = torch.cat([con_c, dis_c], 1).view(-1, 14, 1, 1)
            x_save = self.p(z)
            save_image(x_save.data, './tmp/g_%d_c4.png'%epoch, nrow=10)
        
        #save model
        torch.save([self.q, self.p, self.d],'./model/mnist_cluster.pkl')
