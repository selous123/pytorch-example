import numpy as np
import ctypes
from inception_score_pytorch import inception_score
import torch.utils.data as data
import torch
#author:zhangtao
class cifarDataset(data.Dataset):
    def __init__(self,images):
        self.data = images
    def __getitem__(self,index):
        return self.data[index]
    def __len__(self):
        return len(self.data)


def get_inception_score(nets,batch_size):
    #images with shape [num_samples,3,32,32];
    z = torch.randn(2500,100);
    z = z.cuda()
    images = nets(z)
    dataset = cifarDataset(images)
    return inception_score(dataset,resize=True,splits=10)


def Hungarian(batch_size,c):
    pi = np.zeros([batch_size,batch_size])
    ll = ctypes.cdll.LoadLibrary
    lib = ll("./libhungarian_"+str(batch_size)+".so")
    hungarian_func = lib.func
    hungarian_func.restype = ctypes.POINTER(ctypes.c_int)
    hungarian_func.argtypes = [ctypes.POINTER(ctypes.c_float)]
    c =  c.flatten().tolist()
    c_value =(ctypes.c_float * len(c))(*c)
    result = hungarian_func(c_value)
    for i in range(batch_size):
        #print result[i]
        pi[result[i],i] = 1.0/batch_size
    return pi

def simplexCVX(batch_size,c):
#arguments:
#   batch_size:
#   c         : with shape[batch_size,batch_size]
    ll = ctypes.cdll.LoadLibrary
    lib = ll("./libsimplex_gpu.so")
    #lib.func.argtypes = [ctypes.c_float]
    #set up function
    simplex_func = lib.func
    simplex_func.restype = ctypes.POINTER(ctypes.c_int)
    simplex_func.argtypes = [ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
    #print simplex_func()
    #print '***finish***'
    #batch_size = 10
    #build simplexTableau
    row = 2*batch_size+2
    col = batch_size*batch_size+2*batch_size+1
    A = np.zeros([row,col])
    a1 = np.zeros([batch_size*batch_size,batch_size]);
    a2 = np.zeros([batch_size*batch_size,batch_size])
    for i in range(batch_size):
        a1[batch_size*i:batch_size*(i+1),i] = 1
        a2[batch_size*i:batch_size*(i+1),:] = np.eye(batch_size,batch_size)
    b = np.ones([batch_size*2])/batch_size
    A[:batch_size,:batch_size*batch_size] = a1.T
    A[batch_size:2*batch_size,:batch_size*batch_size] = a2.T
    A[:2*batch_size,batch_size*batch_size:batch_size*batch_size+2*batch_size]=np.eye(2*batch_size,2*batch_size)
    A[:2*batch_size,-1] = b
    A[-1,:batch_size*batch_size] = -2
    A[-1,-1] = -2

    pi = np.zeros(batch_size*batch_size);
    c = c.reshape(batch_size*batch_size)

    A[-2,:batch_size*batch_size] = c
    A1 = A.flatten().tolist()
    simplexTableau = (ctypes.c_float * len(A1))(*A1)
    result = simplex_func(batch_size,simplexTableau)
    for i in range(batch_size):
    	pi[result[i]] = 1.0/batch_size
    pi = pi.reshape(batch_size,-1)
    return pi
