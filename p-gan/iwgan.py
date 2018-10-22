import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
"""
1.weight initialization
2.optimizer(Adam,RMSPro)
3.learning rate
4.parameters
"""
#set parameters
ngf = 128
ndf = 128
nz = 128
nc = 3
cuda = True;
bn = True
dataroot="/home/lrh/dataset/mnist"
batch_size = 32
epoch_num = 200

critic_iters = 5
Lambda = 10
result_directory = "./result_mnist_iwgan_baseline"
env = "iwgan"
win = None


#load pre-trained model
load = False
load_gnet_directory = "/home/lrh/program/git/pytorch-example/cifar-gan/result_iwgan_0528/gnet_290.pkl"
load_dnet_directory = "/home/lrh/program/git/pytorch-example/cifar-gan/result_iwgan_0528/dnet_290.pkl"
#build model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        #input: [batch_size,100,1,1]
        self.deconv0 = nn.ConvTranspose2d(128,256,5,2,0,bias=False)
        self.bn0 = nn.BatchNorm2d(256);
        #class torch.nn.ConvTranspose2d(in_channels, out_channels, ...
        #kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        #ouput_shape[N,64,14,14] 7 = floor((14-5+1+2*padding)/2)
        self.deconv1 = nn.ConvTranspose2d(256,128,5,2,0,bias=False);
        self.bn1 = nn.BatchNorm2d(128);
        self.deconv2 = nn.ConvTranspose2d(128,64,5,2,0,bias=False);
        self.bn2 = nn.BatchNorm2d(64)
        #output_shape[N,1,28,28]
        self.deconv3 = nn.ConvTranspose2d(64,1,5,2,0,bias=False);
        # self.bn3 = nn.BatchNorm(32)
        # self.deconv4 = nn.ConvTranspose2d(32,1,4,2,0,bias=False)


    def forward(self,x):
        #input shape[N,100]
        x = x.unsqueeze(2).unsqueeze(3)
        x = self.deconv0(x)

        x = self.bn0(x)
        x = F.relu(x);
        x = self.deconv1(x);

        x = self.bn1(x);
        x = F.relu(x)
        x = self.deconv2(x);

        x = self.bn2(x);
        x = F.relu(x)

        x = self.deconv3(x);
        x = F.tanh(x);
        #print x.shape
        x = x[ :, :, 17:17 + 28, 17:17+28]
        #output_size=[conf.batch_size,1,28,28]
        return x;

g_net = Generator()
print g_net
# g_net.apply(weights_init)
if cuda:
    g_net = g_net.cuda()
if load:
    g_net.load_state_dict(torch.load(load_gnet_directory))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, ndf, 3, 2, padding=1),
            nn.LayerNorm([ndf,14,14]),
            nn.LeakyReLU(),
            nn.Conv2d(ndf, 2 * ndf, 3, 2, padding=1),
            nn.LayerNorm([2*ndf,7,7]),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(7*7*2*ndf, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 7*7*2*ndf)
        output = self.linear(output)
        return output

d_net = Discriminator()
# d_net.apply(weights_init)
if cuda:
    d_net = d_net.cuda()
if load:
    d_net.load_state_dict(torch.load(load_dnet_directory))

def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    #real_data with shape [batch_size,3,32,32]
    fake_data.requires_grad = True
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement()/batch_size).contiguous().view(batch_size, 1, 28, 28)
    alpha = alpha.cuda() if cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    #print gradient_penalty
    return gradient_penalty


reconstruction_function = nn.MSELoss(size_average=True)
#load dataset

dataset = dset.MNIST(root=dataroot,train = False,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,drop_last=True,shuffle=True)


one = torch.FloatTensor([1])
if cuda:
    one = one.cuda()
mone = one*-1;
## optimizer
d_optimizer = optim.Adam(d_net.parameters(),lr=1e-4,betas=(0.5,0.999))
g_optimizer = optim.Adam(g_net.parameters(),lr=1e-4,betas=(0.5,0.999))

def set_grad(net,bool_value):
    for p in net.parameters():
        p.requires_grad = bool_value
#training
for epoch in range(epoch_num):
    for i,data in enumerate(dataloader,0):
        #optimize discriminator
        set_grad(d_net,True)
        set_grad(g_net,False)
        real_x,labels = data;
        z = torch.randn(batch_size,nz);
        for critic_iter in range(critic_iters):
            d_net.zero_grad()
            if cuda:
                real_x = real_x.cuda()
                z = z.cuda()


            real_labels = d_net(real_x)
            real_label = real_labels.mean()
            real_label.backward(mone)

            fake_x = g_net(z)
            fake_labels = d_net(fake_x)
            fake_label = fake_labels.mean()
            fake_label.backward(one)

            gradient_penalty = calc_gradient_penalty(d_net,real_x,fake_x);
            # fake_ls = d_net(fake_x);
            # fake_l = fake_ls.mean()
            # fake_l.backward()
            gradient_penalty.backward(one)
            d_optimizer.step()

            d_loss = fake_label - real_label + gradient_penalty
        #optimize Generator
        set_grad(g_net,True)
        set_grad(d_net,False)
        g_net.zero_grad()
        z = torch.randn(batch_size,nz);
        if cuda:
            z = z.cuda()
        fake_x = g_net(z)
        fake_labels = d_net(fake_x);
        fake_label = fake_labels.mean()
        fake_label.backward(mone)
        g_optimizer.step()
        g_loss = -fake_label
        mse = reconstruction_function(real_x,fake_x.detach())

        print "epoch is:[{}|{}],index is:[{}|{}],d_loss:{},g_loss:{},mse:{}".\
            format(epoch,epoch_num,i,len(dataloader),d_loss,g_loss,mse);

        z = torch.randn([batch_size,nz]);
        if cuda:
            z = z.cuda()
        fake_x = g_net(z)
        vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png' % (result_directory,epoch),
            normalize=True)

    if epoch%10 == 0:

        torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(result_directory,epoch));
        torch.save(d_net.state_dict(),'%s/dnet_%03d.pkl' %(result_directory,epoch));
