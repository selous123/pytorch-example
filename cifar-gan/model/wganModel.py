import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../');
import config
conf = config.DefaultConf()
class dcgan_generator(nn.Module):
    def __init__(self):
        super(dcgan_generator,self).__init__()
        #input: [batch_size,100,1,1]
        self.deconv0 = nn.ConvTranspose2d(100,256,4,1,0,bias=False)
        self.bn0 = nn.BatchNorm2d(256);
        #class torch.nn.ConvTranspose2d(in_channels, out_channels, ...
        #kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        #ouput_shape[N,64,14,14] 7 = floor((14-5+1+2*padding)/2)
        self.deconv1 = nn.ConvTranspose2d(256,128,4,2,1,bias=False);
        self.bn1 = nn.BatchNorm2d(128);
        self.deconv2 = nn.ConvTranspose2d(128,64,4,2,1,bias=False);
        self.bn2 = nn.BatchNorm2d(64)
        #output_shape[N,3,32,32]
        self.deconv3 = nn.ConvTranspose2d(64,3,4,2,1,bias=False);


    def forward(self,x):
        #input shape[N,100]
        x = self.deconv0(x,output_size=[conf.batch_size,256,4,4])
        if conf.bn:
            x = self.bn0(x)
        x = F.relu(x);

        x = self.deconv1(x,output_size=[conf.batch_size,128,8,8]);
        if conf.bn:
            x = self.bn1(x);
        x = F.relu(x)

        x = self.deconv2(x,output_size=[conf.batch_size,64,16,16]);
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
        if conf.bn:
            x = self.bn1(x)
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
        return x.squeeze();

import numpy as np
def sample(size):
    z = np.random.uniform(-1,1,size=size);
    return z;
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def clip_num(num,max,min):
    if num<min:
        return min;
    elif num>max:
        return max;
    else:
        return num;

def clip_grad(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.grad = clip_num(m.weight.data.grad,-0.01,0.01);
    elif classname.find('BatchNorm') != -1:
        m.weight.data.grad = clip_num(m.weight.data.grad,-0.01,0.01);
        m.bias.data.grad = clip_num(m.bias.data.grad,-0.01,0.01);

if __name__ == '__main__':
    g_net = dcgan_generator();
    g_net = g_net.double()
    g_net.apply(clip_grad)
    # d_net = dcgan_discriminator();
    # d_net = d_net.double()
    #
    # size = conf.batch_size,100;
    # z = sample(size);
    # print z.shape
    # z = torch.from_numpy(z);
    #
    # print g_net
    # fake_x = g_net(z);
    # fake_logits = d_net(fake_x);
    # print fake_logits
    # for name,parameters in g_net.named_parameters():
    #     print name
    #     if name.find('bn') != -1:
    #         print parameters
    #         print "hello"
