from inception_score import inception_score
import torch.utils.data as data
import torch
nz = 128
cuda = True

def cov_function(x):
    dim = x.size[2]
    batch_size = x.size[1];
    cov_matrix = torch.ones(dim,dim)
    if cuda:
        cov_matrix = cov_matrix.cuda()
    for i in range(dim):
        for j in range(dim):
            cov_matrix[i,j] = torch.mul((x[:,i]-x[:,i].mean()),(x[:,j]-x[:,j].mean()))/(batch_size-1)

    return cov_matrix

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
    #ma_distance = torch.mul(torch.matmul(gradients,cov_function(gradients)),gradients)
    #gradient_penalty = ((torch.mean(ma_distance ** 2,dim=2) - 1) ** 2).mean() * Lambda

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class cifarDataset(data.Dataset):
    def __init__(self,images):
        self.data = images
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def get_inception_score(net):
    z = torch.randn(2500,nz)
    if cuda:
        z = z.cuda()
    images = net(z)
    #images with shape [num_samples,3,32,32];
    dataset = cifarDataset(images)
    return inception_score(dataset, cuda=True, batch_size=32, resize=True, splits=10)

def set_grad(net,bool_value):
    for p in net.parameters():
        p.requires_grad = bool_value
