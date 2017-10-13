#coding:utf-8
class DefaultConfig(object):
    def __init__(self):
        ##net configuration
        self.lr = 0.001
        
        ##system configuration
        self.root_path = "/mnt/hgfs/ubuntu14/dataset/cifar-unbalance-data/"
        #self.root_path = "/home/lrh/dataset/cifar-unbalance-data/"
        
        #istraining
        self.istraining = False
        #cuda or not
        self.cuda = False
        
        #self pace learning
        self.K = 1
        self.update_threshold = 0.01

        self.epoch_num = 500
        
if __name__=='__main__':

    config = DefaultConfig()
    print "learning rate is {}".format(config.lr)
    
    