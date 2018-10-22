import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from scipy.optimize import linprog
import torch.nn.functional as F
import utils
import torch.utils.data as data

nz = 100
g_steps = 5
result_directory = "result_lsun_1011_direct_onebatch"
batch_size = 16
bn = True
lr = 3e-4
beta = (0.5,0.999)
epoch_num = 1000000
cuda = True
load = False
load_gnet_directory = "/home/lrh/program/git/pytorch-example/mnist-gan/result_cifar_1009_direct_0.05LAMBDA/gnet_950.pkl"

a1 = np.zeros([batch_size*batch_size,batch_size]);
a2 = np.zeros([batch_size*batch_size,batch_size])
for i in range(batch_size):
    a1[batch_size*i:batch_size*(i+1),i] = 1
    a2[batch_size*i:batch_size*(i+1),:] = np.eye(batch_size,batch_size)
A = np.concatenate((a1,a2),axis=1)
A = A.T
b = np.ones([batch_size*2])/batch_size


class wgan_generator(nn.Module):
    def __init__(self):
        super(wgan_generator,self).__init__()
        #input: [batch_size,100]
        self.fc1 = nn.utils.weight_norm(nn.Linear(nz,2*4*4*1024))

        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.utils.weight_norm(nn.Conv2d(1024,2*512,kernel_size=5,padding=2,stride=1))
        )

        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.utils.weight_norm(nn.Conv2d(512,2*256,kernel_size=5,padding=2,stride=1))
        )

        self.conv3 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.utils.weight_norm(nn.Conv2d(256,2*128,kernel_size=5,padding=2,stride=1))
        )
        self.conv4 = nn.utils.weight_norm(nn.Conv2d(128,3,kernel_size=5,padding=2,stride=1))



    def forward(self,x):

        x = self.fc1(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * F.sigmoid(l) # gated linear unit, one of Alec's tricks
        x = x.view(-1,1024,4,4)
        #[4,4]
        x = self.conv1(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * F.sigmoid(l)
        #[16,16]
        x = self.conv2(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * F.sigmoid(l)
        #[64,,64]
        x = self.conv3(x)
        x,l = torch.chunk(x,2,dim=1)
        x = x * F.sigmoid(l)
        #[256,256]
        x = self.conv4(x)

        x = F.tanh(x)
        return x;

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        diff_norm_1 = diff.norm(1 , dim = 1)
        cost_matrix[:,i] = diff_norm + 0.05 * diff_norm_1
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


class smallCifarDataset(data.Dataset):
    def __init__(self,root_dir):
        self.data = np.load(root_dir)
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return len(self.data)




def train(g_net):
    # data = dset.CIFAR10(root="/home/lrh/dataset/cifar-10",train = False, download=True,transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]))
    #
    # dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True,drop_last=True)
    root_dir = "/home/lrh/program/git/pytorch-example/p-gan/data_small/lsun/image.npy"
    small_dataset = smallCifarDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(small_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

    g_optimizer = optim.Adam(g_net.parameters(),lr=lr,betas = beta)

    for i,data in enumerate(dataloader,0):
        #     #with shape [batch_size,3,32,32]
        real_x = data
        vutils.save_image(real_x,'%s/real_samples.png'
        % (result_directory),normalize=True)
        break;
    for epoch in range(epoch_num):
        z = torch.randn(real_x.size(0),nz);
        #to GPU
        if cuda:
            real_x = real_x.cuda()
            z = z.cuda()

        g_optimizer.zero_grad()
    #
        isopt = True
        pi = None
        for j in range(g_steps):
            fake_x = g_net(z)
            #[batch_size,3,32,32]
            loss,pi = loss_fn(real_x,fake_x,isopt,pi)
            loss.backward()
            g_optimizer.step()
            isopt = False

        if epoch%100==0:
            fake_x = g_net(z)
            vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png'
            % (result_directory,epoch),normalize=True)

            fake_x = fake_x.cpu().detach().numpy()
            print "epoch is:[{}|{}],index is:[{}|{}],g_loss:{}".\
                format(epoch,epoch_num,\
                i,len(dataloader),loss);
        if epoch%10000==0:
            torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(result_directory,epoch));

if __name__=="__main__":

    g_net = wgan_generator()
    g_net.apply(weights_init)
    if cuda:
        g_net.cuda()
    #set moudle.istraing=True
    #net.train()
    if load:
        g_net.load_state_dict(torch.load(load_gnet_directory))
    train(g_net)
