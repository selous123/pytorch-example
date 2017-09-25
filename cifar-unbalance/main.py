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
    loss = -torch.mean(labels*torch.log(logits)+(1-labels)*torch.log(1-logits))

    assert loss.shape==(1,)
    return loss

import torch.optim as optim
from torch.autograd import Variable
def train(net):
    ###read file
    train_dataset = cifarUnbalanceDataset(conf.root_path,train=True)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=512,shuffle=True,drop_last=True)
    #dataiter = iter(dataloader)
    
    ###optimizer
    optimize = optim.SGD(net.parameters(),lr = conf.lr)
    
    for epoch in range(2):
        print "epoch:{}".format(epoch)
        for i,data in enumerate(dataloader,0):
            images,labels = data
            images,labels = Variable(images),Variable(labels)
            #print labels.data.numpy()
            #raw_input("wait")
            logits = net(images)
            #print logits.data.numpy()
            l = loss(logits,labels)
            optimize.zero_grad()
            l.backward()
            optimize.step()
            print "epoch is:{},step is:{},loss is:{}".format(epoch,i,l.data[0])
        
        print "epoch is:{},loss is:{}".format(epoch,l.data[0])

def val(net):
    pass

def test(net):
    #load data
    test_dataset = cifarUnbalanceDataset(conf.root_path,train=False)
    data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=512,shuffle=False,drop_last=False)
    
    total = 0
    correct = 0
    for images,labels in data_loader:
        logits = net(Variable(images))
        ##greater and equal
        predicted = logits.data.ge(0.5)
# =============================================================================
#         print predicted.size()
#         print labels.squeeze().size()
#         raw_input("wait")
# =============================================================================
        labels = labels.type('torch.ByteTensor')
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print "correct:{},total:{}".format(correct,total)
if __name__=='__main__':
    
    net = Net()
    net.double()
    train(net)
    test(net)
    #print "learning rate is {}".format(conf.lr)
