import torch.nn as nn
import torch.nn.functional as F
#z with shape (n_samples,100)
class GeneratorNet(nn.Module):
    def __init__(self):
        super(GeneratorNet,self).__init__()
        self.fc1=nn.Linear(100,128);
        self.fc2=nn.Linear(128,784);
    def forward(self,x):
        x = F.tanh(self.fc1(x));
        x = self.fc2(x);
        x = F.sigmoid(x);
        return x

class DiscriminatorNet(nn.Module):
    def __init__(self):
        super(DiscriminatorNet,self).__init__()
        self.fc1=nn.Linear(784,128);
        self.fc2=nn.Linear(128,1);
    def forward(self,x):
        x = F.tanh(self.fc1(x));
        x = self.fc2(x);
        x = F.sigmoid(x);
        return x

if __name__=='__main__':
    g_nets = GeneratorNet();
    for name,para in g_nets.named_parameters():
        print name
    d_nets = DiscriminatorNet();
