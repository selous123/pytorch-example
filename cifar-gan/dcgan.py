#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.optim as optim
import config
import torchvision.utils as vutils
conf = config.DefaultConf()
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import visutils
import utils

class dcgan_generator(nn.Module):
    def __init__(self):
        super(dcgan_generator,self).__init__()
        #input: [batch_size,100,1,1]
        self.deconv0 = nn.ConvTranspose2d(100,128,4,1,0,bias=False)
        self.bn0 = nn.BatchNorm2d(128);
        #class torch.nn.ConvTranspose2d(in_channels, out_channels, ...
        #kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        #ouput_shape[N,64,16,16] 8 = floor((16-5+1+2*padding)/2)
        self.deconv1 = nn.ConvTranspose2d(128,64,4,2,1,bias=False);
        self.bn1 = nn.BatchNorm2d(64);
        self.deconv2 = nn.ConvTranspose2d(64,32,4,2,1,bias=False);
        self.bn2 = nn.BatchNorm2d(32)
        #output_shape[N,3,32,32]
        self.deconv3 = nn.ConvTranspose2d(32,3,4,2,1,bias=False);


    def forward(self,x):
        #input shape[N,100]
        x = self.deconv0(x,output_size=[conf.batch_size,128,4,4])
        if conf.bn:
            x = self.bn0(x)
        x = F.relu(x);

        x = self.deconv1(x,output_size=[conf.batch_size,64,8,8]);
        if conf.bn:
            x = self.bn1(x);
        x = F.relu(x)

        x = self.deconv2(x,output_size=[conf.batch_size,32,16,16]);
        if conf.bn:
            x = self.bn2(x);
        x = F.relu(x)

        x = self.deconv3(x,output_size=[conf.batch_size,3,32,32]);
        x = F.tanh(x);
        return x;
class dcgan_discriminator(nn.Module):
    def __init__(self):
        super(dcgan_discriminator,self).__init__()
        #input x data[-1,3,32,32,]
        #with shape [N,C_in,H,W];
        self.conv1 = nn.Conv2d(3,32,4,stride=2,padding=1,bias=False);
        self.bn1 = nn.BatchNorm2d(32);
        #self.pool = nn.MaxPool2d(2,stride=2);
        self.conv2 = nn.Conv2d(32,64,4,stride=2,padding=1,bias=False);
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,1,4,1,0,bias=False);

    def forward(self,x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.2)
        #x = self.pool(x);

        x = self.conv2(x);
        if conf.bn:
            x = self.bn2(x);
        x = F.leaky_relu(x,0.2)
        #x = self.pool(x);
        x = self.conv3(x);
        if conf.bn:
            x = self.bn3(x);
        x = F.leaky_relu(x,0.2)
        #(batch_size,1,1,1)
        x = self.conv4(x);
        x = F.sigmoid(x)

        return x.squeeze();
def sample(size):
    '''gaussian prior for G(Z)'''
    z = np.random.normal(size=size).astype(np.float32)
    return torch.from_numpy(z)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def train(d_net,g_net):
    #prepare true samples

    data = dset.CIFAR10(root=conf.root_path, download=False,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    dataloader = torch.utils.data.DataLoader(data,batch_size=conf.batch_size,shuffle=True,drop_last=True)
    #generate fake samples
    d_optimizer = optim.Adam(d_net.parameters(),lr=conf.lr,betas = conf.beta)
    g_optimizer = optim.Adam(g_net.parameters(),lr=conf.lr,betas = conf.beta)
    d_lo = g_lo = 0;
    criterion = nn.BCELoss()
    if conf.fixz:
        size = conf.batch_size,100,1,1
        z = sample(size);
    for epoch in range(conf.epoch_num):
        for i,data in enumerate(dataloader,0):
            #with shape [batch_size,3,32,32]
            real_x = data[0]

            #to GPU
            if conf.cuda:
                real_x = real_x.cuda()


            d_optimizer.zero_grad()
            num_samples = real_x.size(0);
            label = torch.full((num_samples,), 1).cuda();
            real_logits = d_net(real_x);
            d_l_r = criterion(real_logits,label);
            d_l_r.backward()
            if not conf.fixz:
                size = num_samples,100,1,1
                z = sample(size);
            if conf.cuda:
                z = z.cuda()
            fake_x = g_net(z);
            label.fill_(0)
            fake_logits = d_net(fake_x)
            d_l_f = criterion(fake_logits,label);
            d_l_f.backward()
            d_optimizer.step()
            d_lo = d_l_r+d_l_f


            g_optimizer.zero_grad()
            fake_x = g_net(z)
            fake_logits = d_net(fake_x)
            label.fill_(1)
            g_l = criterion(fake_logits,label);
            g_l.backward();
            g_optimizer.step()
            g_lo = g_l

            if conf.debug:
                print "epoch is：[{}|{}]，index is :[{}|{}],d_loss:{},g_loss:{}".\
                format(epoch,conf.epoch_num,\
                i,len(dataloader),d_lo,g_lo);
        #after each epoch,we visulize the result
        if conf.debug:
            for para in g_net.parameters():
                print torch.mean(para.grad)
        fake_x = g_net(z)
        print "d_loss:{},g_loss:{}".format(d_lo,g_lo)
        vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png'
        % (conf.result_directory,epoch),normalize=True)


        inception_scores = utils.get_inception_score(g_net);
        inception_score = np.array([inception_scores[0]])
        print inception_score
        conf.win = visutils.visualize_loss(epoch,inception_score,conf.env,conf.win)


        if epoch%50==0:
            torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(conf.result_directory,epoch));
            torch.save(d_net.state_dict(),'%s/dnet_%03d.pkl' %(conf.result_directory,epoch));


if __name__=='__main__':

# from model.model import GeneratorNet
# from model.model import DiscriminatorNet
    # d_net = DiscriminatorNet()
    # g_net = GeneratorNet()
    d_net = dcgan_discriminator()
    g_net = dcgan_generator()
    d_net.apply(weights_init)
    g_net.apply(weights_init)
    if conf.cuda:
        d_net.cuda()
        g_net.cuda()
    if conf.istraining:
        #set moudle.istraing=True
        #net.train()
        train(d_net,g_net)
