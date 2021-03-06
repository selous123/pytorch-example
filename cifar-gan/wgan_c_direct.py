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


class classify_net(nn.Module):

    def __init__(self):
        super(classify_net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5,padding=(2,2))
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5,padding=(2,2))
        self.bn2 = nn.BatchNorm2d(16)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

c_net = classify_net().cuda()
pkl_dir = "/home/lrh/program/git/pytorch-example/mnist_cnn/pkls/mnist_init_300epoch.pkl"
c_net.load_state_dict(torch.load(pkl_dir))

class wgan_generator(nn.Module):
    def __init__(self):
        super(wgan_generator,self).__init__()
        #input: [batch_size,100,1,1]
        self.deconv0 = nn.ConvTranspose2d(100,256,5,2,0,bias=False)
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
        x = self.deconv0(x)
        if conf.bn:
            x = self.bn0(x)
        x = F.relu(x);
        x = self.deconv1(x);
        if conf.bn:
            x = self.bn1(x);
        x = F.relu(x)
        x = self.deconv2(x);
        if conf.bn:
            x = self.bn2(x);
        x = F.relu(x)
        #output_size=[conf.batch_size,1,28,28]
        x = self.deconv3(x);
        x = F.tanh(x);
        #print x.shape
        x = x[ :, :, 17:17 + 28, 17:17+28]

        return x;

# class wgan_generator(nn.Module):
#     def __init__(self):
#         super(wgan_generator,self).__init__()
#         self.fc1 = nn.Linear(100,1024)
#         self.bn1 = nn.BatchNorm1d(1024)
#         self.fc2 = nn.Linear(1024,7*7*128)
#         self.bn2 = nn.BatchNorm1d(7*7*128)
#         self.deconv1 = nn.ConvTranspose2d(128,64,4,2,1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.deconv2 = nn.ConvTranspose2d(64,1,4,2,1)
#
#     def forward(self,x):
#         x = x.squeeze()
#         #[32,100]
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         #[32,1024]
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.relu(x)
#         #[32,7*7*128]
#         x = x.reshape([32,128,7,7])
#         #[32,128,7,7]
#         x = self.deconv1(x)
#         x = self.bn3(x)
#         x = F.relu(x)
#         #[32,64,14,14]
#         x = self.deconv2(x)
#         #[32,1,28,28]
#         x = F.sigmoid(x)
#         return x




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def loss_fn(real_x,fake_x,pi=None):
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
    loss = torch.sum(torch.mul(cost_matrix,pi))
    return loss

def calc_label_penalty(real_feas,fake_feas,isopt=True,pi=None):
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
        #min_value,min_pos = torch.min(diff_norm,dim=0)
        #print min_pos
        #dist_matrix[min_pos,i] = 1.0/batch_size
    #[batch_size,batch_size]
    if isopt:
        c = cost_matrix.cpu().detach().numpy().reshape(batch_size*batch_size)
        res = linprog(c, A_eq=A, b_eq=b,options={'maxiter':5000})
        pi = torch.from_numpy(res.x.reshape(batch_size,batch_size)).float().cuda()
        #print pi[0,:]
    #assert(type(pi)!=Nonetype)
    label_penalty = torch.sum(torch.mul(cost_matrix,pi)) * Lambda

    return label_penalty,pi

def train(g_net):
    # data = dset.CIFAR10(root=conf.root_path, download=False,transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]))
    data = dset.MNIST(root=conf.root_path,train = False,download=True,transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
            z = torch.randn(real_x.size(0),100,1,1);
            #to GPU
            if conf.cuda:
                real_x = real_x.cuda()
                z = z.cuda()

            g_optimizer.zero_grad()
            c_net.zero_grad()

            isopt = True
            pi = None
            for j in range(conf.g_steps):
                fake_x = g_net(z)

                #calc label penalty
                real_feas = c_net(real_x)
                fake_feas = c_net(fake_x)
                label_penalty,pi = calc_label_penalty(real_feas,fake_feas,isopt,pi)
                label_penalty.backward(retain_graph=True)

                #calc norm loss
                loss = loss_fn(real_x,fake_x,pi)
                loss.backward()

                #update gnet
                g_optimizer.step()
                isopt = False
            g_loss = label_penalty + loss
            print "epoch is:[{}|{}],index is:[{}|{}],g_loss:{},label_penalty:{},loss:{}".\
                format(epoch,conf.epoch_num,\
                i,len(dataloader),g_loss,label_penalty,loss);

        z = torch.randn(64,100,1,1)
        z = z.cuda()
        fake_x = g_net(z)
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
