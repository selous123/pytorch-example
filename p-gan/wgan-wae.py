import torch
import torch.nn as nn
import config
conf = config.DefaultConf()
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from scipy.optimize import linprog
import torch.nn.functional as F
import utils

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.n_channel = 1
        self.dim_h = 128
        self.n_z = 8

        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.main(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.n_channel = 1
        self.dim_h = 128
        self.n_z = 8

        self.proj = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU()
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.main(x)
        return x


encoder = Encoder().cuda()
pkl_dir = "/home/lrh/program/git/pytorch-example/mnist_autoencoder/wae_mmd/wae_encoder.pth"
encoder.load_state_dict(torch.load(pkl_dir))
encoder.require_grad = False
print encoder

decoder = Decoder().cuda()
pkl_dir = "/home/lrh/program/git/pytorch-example/mnist_autoencoder/wae_mmd/wae_decoder.pth"
decoder.load_state_dict(torch.load(pkl_dir))
decoder.require_grad = False
print decoder

# z = torch.randn(128,8)
# z = z.cuda()
# fake_x = decoder(z)
# print fake_x.shape
# vutils.save_image(fake_x.cpu().detach(),'fake_samples_epoch_1.png',normalize=True)

class wgan_generator(nn.Module):
    def __init__(self):
        super(wgan_generator,self).__init__()
        self.nz = 100
        self.dim_h = 128
        self.oz = 8
        self.fc1 = nn.Linear(self.nz,self.dim_h)
        self.fc2 = nn.Linear(self.dim_h,self.dim_h*2)
        self.fc3 = nn.Linear(self.dim_h*2,self.dim_h*4)
        self.fc4 = nn.Linear(self.dim_h*4,self.dim_h*2)
        self.fc5 = nn.Linear(self.dim_h*2,self.oz)

    def forward(self,x):
        x = self.fc1(x)
        #x = F.relu(x)
        x = self.fc2(x)
        #x = F.relu(x)
        x = self.fc3(x)
        #x = F.sigmoid(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


def loss_fn(real_x,fake_x,isopt=True,pi=None):
    batch_size = real_x.size(0)
    cost_matrix = torch.zeros([batch_size,batch_size]).cuda()
    dist_matrix = np.zeros([batch_size,batch_size])
    for i in range(batch_size):
        diff = real_x - fake_x[i]
        diff = diff.view(diff.size(0),-1)
        #||x_i - y_j||_2
        #[batch_size,1]
        diff_norm = diff.norm(2 , dim = 1)
        cost_matrix[:,i] = diff_norm
        #min_value,min_pos = torch.min(diff_norm,dim=0)
        #print min_pos
        #dist_matrix[min_pos,i] = 1.0/batch_size
    #[batch_size,batch_size]
    if isopt:
        c = cost_matrix.cpu().detach().numpy().reshape(batch_size*batch_size)
        pi = utils.simplexCVX(batch_size,c);
        pi = torch.from_numpy(pi).float().cuda()
        #print pi[0,:]
    #assert(type(pi)!=Nonetype)
    loss = torch.sum(torch.mul(cost_matrix,pi))
    return loss,pi

def train(g_net):
    # data = dset.CIFAR10(root=conf.root_path, download=False,transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]))
    data = dset.MNIST(root=conf.root_path,train = False,download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    dataloader = torch.utils.data.DataLoader(data,batch_size=conf.batch_size,shuffle=True,drop_last=True)
#     #generate fake samples
    g_optimizer = optim.Adam(g_net.parameters(),lr=conf.lr,betas = conf.beta)
#     #not run
#     #c_optimizer = optim.Adam(c_net.parameters(),lr=0)
    for epoch in range(conf.epoch_num):
        for i,data in enumerate(dataloader,0):
            #with shape [batch_size,3,32,32]
            real_x = data[0]
            z = torch.randn(real_x.size(0),100);
            #to GPU
            if conf.cuda:
                real_x = real_x.cuda()
                z = z.cuda()
            g_optimizer.zero_grad()

            fake_feas = g_net(z)
            #print "fake:{}".format(fake_logvar)
            #calc label penalty
            real_feas = encoder(real_x)
            # real_z = c_net.reparametrize(real_mu,real_logvar)
            #fake_x = decoder(real_feas)
            # fake_x = fake_x.reshape(32,1,28,28)
            #print real_logvar
            g_loss,pi= loss_fn(real_feas,fake_feas)
            g_loss.backward()

            #update gnet
            g_optimizer.step()

            print "epoch is:[{}|{}],index is:[{}|{}],g_loss:{}".\
                format(epoch,conf.epoch_num,i,len(dataloader),g_loss);

        z = torch.randn(128,100)
        z = z.cuda()
        fake_feas = g_net(z)
        fake_x = decoder(fake_feas)
        vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png'
        % (conf.result_directory,epoch),normalize=True)
        if epoch%50==0:
            torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(conf.result_directory,epoch));
        #
#
if __name__=="__main__":
    g_net = wgan_generator()
    g_net = g_net.cuda()
    if conf.cuda:
        g_net.cuda()
    if conf.istraining:
        #set moudle.istraing=True
        #net.train()
        train(g_net)
