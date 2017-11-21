#coding:utf-8
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from config import DefaultConf
conf = DefaultConf()
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
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
        if conf.bn:
            x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        if conf.bn:
            x = self.bn2(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x,dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
if __name__=="__main__":
    net = Net()
    print net  
