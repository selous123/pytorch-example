import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.autograd as autograd
import visutils
import utils
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
dataroot="/home/lrh/dataset/cifar-10"
batch_size = 64
epoch_num = 300

critic_iters = 5
Lambda = 10
result_directory = "./result_iwgan_layernorm_0601"
env = "iwgan"
win = None
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
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * ngf),
            #nn.BatchNorm2d(4 * 4 * 4 * ngf),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 2, stride=2),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * ngf, ngf, 2, stride=2),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(ngf, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * ngf, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator,self).__init__()
#         self.deconv1 = nn.ConvTranspose2d(nz,ndf*4,4,1,0,bias=False)
#         self.bn1 = nn.BatchNorm2d(ndf*4)
#         #[batch_size,ngf*4,4,4]
#         self.deconv2 = nn.ConvTranspose2d(ndf*4,ndf*2,4,2,1,bias=False)
#         self.bn2 = nn.BatchNorm2d(ndf*2)
#         #[batch_size,ngf*2,8,8]
#         self.deconv3 = nn.ConvTranspose2d(ndf*2,ndf*1,4,2,1,bias=False)
#         self.bn3 = nn.BatchNorm2d(ndf)
#         #[batch_size,ngf*1,16,16]
#         self.deconv4 = nn.ConvTranspose2d(ndf,nc,4,2,1,bias=False)
#         #[batch_size,3,32,32]
#     def forward(self,x):
#         #input x with shape[batch_size,nz,1,1]
#         x = self.deconv1(x,output_size=[batch_size,ndf*4,4,4])
#         if bn:
#             x = self.bn1(x)
#         x = F.relu(x)
#         x = self.deconv2(x,output_size=[batch_size,ndf*2,8,8])
#         if bn:
#             x = self.bn2(x)
#         x = F.relu(x)
#         x = self.deconv3(x,output_size=[batch_size,ndf*1,16,16])
#         if bn:
#             x = self.bn3(x)
#         x = F.relu(x)
#         x = self.deconv4(x,output_size=[batch_size,nc,32,32])
#         x = F.tanh(x)
#         #return [batch_size,nc,32,32]
#         return x

g_net = Generator()
print g_net
# g_net.apply(weights_init)
if cuda:
    g_net = g_net.cuda()

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator,self).__init__()
#         #input x data[-1,3,32,32,]
#         #with shape [N,C_in,H,W];
#         self.conv1 = nn.Conv2d(nc,ngf,4,stride=2,padding=1,bias=False);
#         #self.bn1 = nn.BatchNorm2d(ngf);
#         #self.pool = nn.MaxPool2d(2,stride=2);
#         self.conv2 = nn.Conv2d(ngf,ngf*2,4,stride=2,padding=1,bias=False);
#         #self.bn2 = nn.BatchNorm2d(ngf*2)
#         self.conv3 = nn.Conv2d(ngf*2,ngf*4,4,stride=2,padding=1,bias=False)
#         #self.bn3 = nn.BatchNorm2d(ngf*4)
#         self.conv4 = nn.Conv2d(ngf*4,1,4,1,0,bias=False);
#     def forward(self,x):
#         #input x with shape[batch_size,3,32,32]
#         x = self.conv1(x)
#         #if bn:
#         #    x = self.bn1(x)
#         x = F.leaky_relu(x,0.2)
#         #x = self.pool(x);
#
#         x = self.conv2(x);
#         #if bn:
#         #    x = self.bn2(x);
#         x = F.leaky_relu(x,0.2)
#         #x = self.pool(x);
#         x = self.conv3(x);
#         #if bn:
#         #    x = self.bn3(x);
#         x = F.leaky_relu(x,0.2)
#         #(batch_size,1,1,1)
#         x = self.conv4(x);
#         return x.squeeze();
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, ndf, 3, 2, padding=1),
            nn.LayerNorm([ndf,16,16]),
            nn.LeakyReLU(),
            nn.Conv2d(ndf, 2 * ndf, 3, 2, padding=1),
            nn.LayerNorm([2*ndf,8,8]),
            nn.LeakyReLU(),
            nn.Conv2d(2 * ndf, 4 * ndf, 3, 2, padding=1),
            nn.LayerNorm([4*ndf,4,4]),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear = nn.Linear(4*4*4*ndf, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*ndf)
        output = self.linear(output)
        return output

d_net = Discriminator()
# d_net.apply(weights_init)
if cuda:
    d_net = d_net.cuda()


def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    #real_data with shape [batch_size,3,32,32]
    fake_data.requires_grad = True
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, real_data.nelement()/batch_size).contiguous().view(batch_size, 3, 32, 32)
    alpha = alpha.cuda() if cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty

#load dataset

dataset = dset.CIFAR10(root=dataroot,download=False,transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
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

            fake_x = g_net(z)
            real_labels = d_net(real_x)
            real_label = real_labels.mean()
            real_label.backward(mone)

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
        print "epoch is:[{}|{}],index is:[{}|{}],d_loss:{},g_loss:{}".\
            format(epoch,epoch_num,i,len(dataloader),d_loss,g_loss);

    #visulize inception score
    inception_scores = utils.get_inception_score(g_net);
    inception_score = np.array([inception_scores[0]])
    win = visutils.visualize_loss(epoch,inception_score,env,win)



    if epoch%10 == 0:
        z = torch.randn([batch_size,nz]);
        if cuda:
            z = z.cuda()
        fake_x = g_net(z)
        vutils.save_image(fake_x.cpu().detach(),'%s/fake_samples_epoch_%03d.png' % (result_directory,epoch),
            normalize=True)
        torch.save(g_net.state_dict(),'%s/gnet_%03d.pkl' %(result_directory,epoch));
        torch.save(d_net.state_dict(),'%s/dnet_%03d.pkl' %(result_directory,epoch));
