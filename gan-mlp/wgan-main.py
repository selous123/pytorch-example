#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.optim as optim
from data.dataset import mnistData
import config
conf = config.DefaultConf()
import numpy as np

def sample(size):
    '''Uniform prior for G(Z)'''
    z = np.random.uniform(-1., 1., size=size)
    return torch.from_numpy(z)

def d_loss(fake_logits,real_logits):
    r_loss = torch.mean(real_logits)
    f_loss = torch.mean(fake_logits)
    loss = -(r_loss - f_loss)
    return loss;
def g_loss(fake_logits):
    loss = -torch.mean(fake_logits)
    return loss



# def visualize_result(fake_x,epoch):
#     fake_images = fake_x.reshape(-1,28,28);
#     img_dir = str(epoch)
#     import os
#     import cv2
#     import shutil
#     if os.path.exists('result/'+img_dir):
#         shutil.rmtree('result/'+img_dir)
#     os.mkdir('result/'+img_dir);
#     for ind in range(fake_images.shape[0]):
#         cv2.imwrite('result/'+img_dir+'/'+str(ind)+'.jpg',fake_images[ind,:,:]*255);
def visualize_result(fake_x,epoch):
    fake_images = fake_x.reshape(-1,28,28);
    import cv2
    #h = int(np.sqrt(conf.epoch_num));
    h = 8
    height = h * 28;
    sum_a = np.ones([height,height]);
    index = 0;
    for i in range(h):
        for j in range(h):
            sum_a[i*28:(i+1)*28,j*28:(j+1)*28] = fake_images[index,:,:];
            index+=1;

    cv2.imwrite(conf.result_directory+str(epoch)+'.jpg',sum_a*255);

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def clip_num(num,min,max):
    max = torch.Tensor([max]).cuda()
    min = torch.Tensor([min]).cuda()
    num = torch.where(num>max,max,num)
    num = torch.where(num<min,min,num)
    return num

def clip_grad(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.grad = clip_num(m.weight.grad,-0.01,0.01);
    elif classname.find('BatchNorm') != -1:
        m.weight.grad = clip_num(m.weight.grad,-0.01,0.01);
        m.bias.grad = clip_num(m.bias.grad,-0.01,0.01);

def train(d_net,g_net):
    #prepare true samples
    data = mnistData(conf.root_path)
    dataloader = torch.utils.data.DataLoader(data,batch_size=conf.batch_size,drop_last=True)
    #generate fake samples
    size = conf.batch_size,100,1,1
    z = sample(size);

    d_optimizer = optim.RMSprop(d_net.parameters(),lr=conf.lr)
    g_optimizer = optim.RMSprop(g_net.parameters(),lr=conf.lr)

    d_lo = g_lo = 0;
    for epoch in range(conf.epoch_num):
        for i,data in enumerate(dataloader,0):
            real_x = data
            if conf.method == 0:
                real_x = real_x.reshape(-1,28*28)
            elif conf.method == 1:
                real_x = real_x.reshape(-1,1,28,28);
            #to GPU
            if conf.cuda:
                real_x,z = real_x.float().cuda(),z.float().cuda()

            #train discriminator
            for d_step in range(conf.d_steps):
                fake_x = g_net(z);
                fake_logits = d_net(fake_x)
                real_logits = d_net(real_x)
                #print torch.min(real_logits)
                #print torch.max(fake_logits)
                d_l = d_loss(fake_logits,real_logits);
                d_optimizer.zero_grad()
                d_l.backward();
                d_net.apply(clip_grad)
                #for para in d_net.parameters():
                    #print para.grad
                d_optimizer.step()
                d_lo = d_l
            #train generator
            for g_step in range(conf.g_steps):
                fake_x = g_net(z);
                fake_logits = d_net(fake_x)
                g_l = g_loss(fake_logits);
                g_optimizer.zero_grad()
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
        tmp_fake_x = fake_x.cpu().detach().numpy()
        if conf.method==1:
            tmp_fake_x = tmp_fake_x.squeeze();
        visualize_result(tmp_fake_x,epoch)
    torch.save(gnet.state_dict(),conf.result_directory+"gnet.pkl");
    torch.save(dnet.state_dict(),conf.result_directory+"dnet.pkl");


if __name__=='__main__':

# from model.model import GeneratorNet
# from model.model import DiscriminatorNet
    # d_net = DiscriminatorNet()
    # g_net = GeneratorNet()
    from model.wganModel import dcgan_generator
    from model.wganModel import dcgan_discriminator
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
