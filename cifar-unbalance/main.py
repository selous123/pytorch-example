from config import DefaultConfig
from models.BasicModel import Net
from data.dataset import cifarUnbalanceDataset
import numpy as np
import torch

conf = DefaultConfig()

def loss(logits,labels):
    """
    Args:
        logits: prediction of samples,[batch_size,1]
        labels: true label of samples,[batch_size,1]
    Return:
        cross entropy loss 
    """
    loss = torch.mean(labels*torch.log(logits)+(1-labels)*torch.log(1-logits))

    assert loss.shape==(1,)
    return loss

import torch.optim as optim
from torch.autograd import Variable
def train(net):
    ###read file
    train_dataset = cifarUnbalanceDataset(conf.root_path,train=False)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=128,shuffle=True,drop_last = True)
    dataiter = iter(dataloader)
    
    ###optimizer
    optimize = optim.SGD(net.parameters(),lr = conf.lr)
    
    for epoch in range(10):
        print "epoch:{}".format(epoch)
        for images,labels in dataiter:
            images,labels = Variable(images),Variable(labels)
            logits = net(images)
            l = loss(logits,labels)
            optimize.zero_grad()
            l.backward()
            optimize.step()
        
        print "epoch is:{},loss is:{}".format(epoch,l)
    

if __name__=='__main__':
    
    net = Net()
    net.double()
    train(net)
    
    print "learning rate is {}".format(conf.lr)
