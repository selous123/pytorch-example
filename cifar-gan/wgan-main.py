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



def sample(size):
    '''gaussian prior for G(Z)'''
    z = np.random.normal(size=size).astype(np.float32)
    return torch.from_numpy(z)

def d_loss(fake_logits,real_logits):
    r_loss = torch.mean(real_logits)
    f_loss = torch.mean(fake_logits)
    loss = -(r_loss - f_loss)
    return loss;
def g_loss(fake_logits):
    loss = -torch.mean(fake_logits)
    return loss

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

            #train discriminator
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
        if epoch%50==0:
            torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(conf.result_directory,epoch));
            torch.save(d_net.state_dict(),'%s/dnet_%03d.pkl' %(conf.result_directory,epoch));


if __name__=='__main__':

# from model.model import GeneratorNet
# from model.model import DiscriminatorNet
    # d_net = DiscriminatorNet()
    # g_net = GeneratorNet()
    from model.dcganModel import dcgan_generator
    from model.dcganModel import dcgan_discriminator
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
