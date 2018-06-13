#
import torch.nn as nn
import torch.optim as optim
#build Module
class ResidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,stride,1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,1,1,bias=False),
            nn.BatchNorm2d(out_channel)
        )

        self.right=shortcut

    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)



class Resnet_Generator(nn.Module):
    def __init__(self):
        super(Resnet_Generator,self).__init__()
        pass

    def forward(self,x):
        pass


class Resnet_Discriminator(nn.Module):
    def __init__(self):
        super(Resnet_Discriminator,self).__init__()
        pass
    def forward(self,x):
        pass


d_optimizer = optim.Adam(d_net.parameters(),lr = 2e-4,betas = (0.5,0.99));
g_optimizer = optim.Adam(g_net.parameters(),lr = 2e-4,betas = (0.5,0.99));
