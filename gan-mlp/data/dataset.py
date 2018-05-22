#coding:utf-8

import torch.utils.data as data
import input_mnist_data as input_data
import numpy as np

def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = np.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


##read the whole file
class mnistData(data.Dataset):
    """
    init dataset class
    Args  :
        root : root path
    Return:
        self.root,self.data,self.labels

    """
    def __init__(self,root,augment=True):
        self.root = root
        ##true sample dataset
        self.data,self.labels = input_data.read_all_data(self.root)
        if augment:
            self.data = self.data / 255.0
        self.data = self.data.astype(np.float)
        self.labels = self.data.astype(np.float)


    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
