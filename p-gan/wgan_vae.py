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

Lambda = 10

a1 = np.zeros([conf.batch_size*conf.batch_size,conf.batch_size]);
a2 = np.zeros([conf.batch_size*conf.batch_size,conf.batch_size])
for i in range(conf.batch_size):
    a1[conf.batch_size*i:conf.batch_size*(i+1),i] = 1
    a2[conf.batch_size*i:conf.batch_size*(i+1),:] = np.eye(conf.batch_size,conf.batch_size)
A = np.concatenate((a1,a2),axis=1)
A = A.T
b = np.ones([conf.batch_size*2])/conf.batch_size


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        x = x.reshape(x.size(0),-1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


c_net = VAE().cuda()
pkl_dir = "/home/lrh/program/git/pytorch-example/mnist_autoencoder/vae.pth"
c_net.load_state_dict(torch.load(pkl_dir))
c_net.require_grad = False

# data = dset.MNIST(root=conf.root_path,train = True,download=True,transform=transforms.Compose([
#             transforms.ToTensor(),
#             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ]))
# dataloader = torch.utils.data.DataLoader(data,batch_size=conf.batch_size,shuffle=True,drop_last=True)
#
#
# for epoch,data in enumerate(dataloader,0):
#     real_x,label = data
#     real_x = real_x.cuda()
#     fake_x,mu,var = c_net(real_x)
#     fake_x = fake_x.reshape(32,1,28,28)
#     vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png'
#     % (conf.result_directory,epoch),normalize=True)
#     exit(0)
# fake_mu = torch.randn(64,20).cuda()
# #fake_sigma = torch.randn(64,20).cuda()
#
# #fake_x = c_net.decode(c_net.reparametrize(fake_mu,fake_sigma))
# fake_x = c_net.decode(fake_mu)
# fake_x = fake_x.reshape(64,1,28,28)
# vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png'
# % (conf.result_directory,0),normalize=True)
#
# exit(0)


class wgan_generator(nn.Module):
    def __init__(self):
        super(wgan_generator,self).__init__()
        #input: [batch_size,100,1,1]

        self.fc1 = nn.Linear(100,400)
        self.fc21 = nn.Linear(400,20)
        self.fc22 = nn.Linear(400,20)
    def forward(self,x):
        #input shape[N,100]
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu,logvar;



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def calc_label_penalty(real_feas,fake_feas,pi=None):
    #min \sum_ij || real_feas_i - fake_feas_j ||_2^2 * \pi_{ij}
    #s.t. \sum_i \pi_{ij} = 1/n
    #     \sum_j \pi_{ij} = 1/n
    #     \pi_{ij}>=0
    batch_size = real_feas.size(0)
    cost_matrix = torch.zeros([batch_size,batch_size]).cuda()
    dist_matrix = np.zeros([batch_size,batch_size])
    for i in range(batch_size):
        diff = real_feas - fake_feas[i]
        diff = diff.view(diff.size(0),-1)
        #||x_i - y_j||_2
        #[batch_size,1]
        diff_norm = diff.norm(2 , dim = 1)
        cost_matrix[:,i] = diff_norm
    #[batch_size,batch_size]
    if pi is None:
        c = cost_matrix.cpu().detach().numpy().reshape(batch_size*batch_size)
        res = linprog(c, A_eq=A, b_eq=b,options={'maxiter':5000})
        pi = torch.from_numpy(res.x.reshape(batch_size,batch_size)).float().cuda()
    label_penalty = torch.sum(torch.mul(cost_matrix,pi)) * Lambda
    return label_penalty,pi

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
    #generate fake samples
    g_optimizer = optim.Adam(g_net.parameters(),lr=conf.lr,betas = conf.beta)
    #not run
    #c_optimizer = optim.Adam(c_net.parameters(),lr=0)
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


            fake_mu,fake_logvar = g_net(z)
            #print "fake:{}".format(fake_logvar)
                #calc label penalty
            real_mu,real_logvar = c_net.encode(real_x)
            # real_z = c_net.reparametrize(real_mu,real_logvar)
            # fake_x = c_net.decode(real_z)
            # fake_x = fake_x.reshape(32,1,28,28)
            #print real_logvar
            logvar_penalty,pi = calc_label_penalty(real_logvar,fake_logvar)
            logvar_penalty.backward(retain_graph=True)

            mu_penalty,pi = calc_label_penalty(real_mu,fake_mu,pi)
            mu_penalty.backward()


            #update gnet
            g_optimizer.step()
            g_loss = logvar_penalty + mu_penalty

            print "epoch is:[{}|{}],index is:[{}|{}],g_loss:{},logvar:{},mu:{}".\
                format(epoch,conf.epoch_num,i,len(dataloader),g_loss,logvar_penalty,mu_penalty);


        z = torch.randn(64,100)
        z = z.cuda()
        fake_mu,fake_var = g_net(z)
        fake_x = c_net.decode(c_net.reparametrize(fake_mu,fake_var))
        fake_x = fake_x.reshape(64,1,28,28)
        vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png'
        % (conf.result_directory,epoch),normalize=True)
        if epoch%50==0:
            torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(conf.result_directory,epoch));


if __name__=="__main__":

    g_net = wgan_generator()
    g_net.apply(weights_init)
    if conf.cuda:
        g_net.cuda()
    if conf.istraining:
        #set moudle.istraing=True
        #net.train()
        train(g_net)
