#coding:utf-8
class DefaultConfig(object):
    def __init__(self):
        ##net configuration
        self.lr = 0.1
        
        ##system configuration
        self.root_path = "/mnt/hgfs/ubuntu14/dataset/cifar-unbalance-data/"


if __name__=='__main__':

    config = DefaultConfig()
    print "learning rate is {}".format(config.lr)