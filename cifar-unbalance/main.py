from config import DefaultConfig
from models.BasicModel import Net
from data.dataset import cifarUnbalanceDataset
import numpy as np
import torch

conf = DefaultConfig()

def loss(logits,labels,*args):
    """
    Args:
        logits: prediction of samples,[batch_size,1]
        labels: true label of samples,[batch_size,1]
    Return:
        cross entropy loss,[1,]
    """
    if args:
        logits = logits[args[0]]
        labels = labels[args[0]]
    loss = -torch.mean(labels*torch.log(logits)+(1-labels)*torch.log(1-logits))
    assert loss.size()==(1,)
    return loss
    """
    loss = labels*torch.log(logits)+(1-labels)*torch.log(1-logits)
    v = loss.le(1/conf.K)
    loss = -torch.mean(loss[v])
    """

def computeV(labels,logits):
    
    loss = -(labels*torch.log(logits)+(1-labels)*torch.log(1-logits))
    #if loss<1/K : v = 1"easy sample"
    v = torch.le(loss,1/conf.K)
    return v

import torch.optim as optim
from torch.autograd import Variable
def train(net):
    ###read file
    train_dataset = cifarUnbalanceDataset(conf.root_path,train=True)
    dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=512,shuffle=True,drop_last=True)
    #dataiter = iter(dataloader)
    
    ###optimizer
    optimize = optim.SGD(net.parameters(),lr = conf.lr)
    
    for epoch in range(conf.epoch_num):
        #each epoch loss
        running_loss = []
        #print "epoch:{}".format(epoch)
        for i,data in enumerate(dataloader,0):
            images,labels = data
            
            images,labels = Variable(images.cuda()),Variable(labels.cuda())
            #print labels.data.numpy()
            #raw_input("wait")
            logits = net(images)
            
            #choose "easy" sample
            #v = computeV(logits,labels)
            #print logits.data.numpy()
            l = loss(logits,labels)
            #running_loss.append(l.cpu().data().numpy())
            optimize.zero_grad()
            l.backward()
            optimize.step()
            #print "epoch is:{},step is:{},loss is:{}".format(epoch,i,l.data[0])
        #print running_loss
        #if sum(running_loss)/len(running_loss) < conf.update_threshold:
        #   conf.K = conf.K/2
            
        print "epoch is:{},loss is:{}".format(epoch,l.data[0])

def val(net):
    pass

def test(net):
    #load data
    test_dataset = cifarUnbalanceDataset(conf.root_path,train=False)
    data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=512,shuffle=False,drop_last=False)
    
    true_positive_num = 0
    predicted_positive_num = 0
    predicted_true_positive_num = 0
    for images,labels in data_loader:
        #move the images to GPU
        if conf.cuda:
            images = images.cuda()
        logits = net(Variable(images))
        ##greater and equal
        #print logits
        predicted = logits.data.ge(0.5)
        if conf.cuda:
            predicted = predicted.cpu()
        
        predicted = predicted.numpy() 
        #print predicted
        labels = labels.numpy()
        true_positive_num += np.sum(labels == 1)
        predicted_positive_num += np.sum(predicted == 1)
        predicted_true_positive_num += np.sum((labels==1)&(predicted==1))
# =============================================================================
#         print predicted.size()
#         print labels.squeeze().size()
#         raw_input("wait")
#
#        #labels = labels.type('torch.ByteTensor')
#        total += labels.size(0)
#        correct += (predicted.cpu().numpy() == labels.numpy()).sum()
# =============================================================================
    print "true_positive_num:{},predicted_positive_num:{},predicted_true_positive_num:{}"\
        .format(true_positive_num,predicted_positive_num,predicted_true_positive_num)
if __name__=='__main__':
    
    
    net = Net()
    if conf.cuda:
        net.cuda()
    net.double()
    if conf.istraining:
        train(net)
        torch.save(net.state_dict(),"init.pkl")
    else:
        net.load_state_dict(torch.load("init.pkl",map_location=lambda storage, loc: storage))
        test(net)
    #print "learning rate is {}".format(conf.lr)
