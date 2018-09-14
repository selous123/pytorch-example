import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, reshape = True ,window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.reshape = reshape

    def forward(self, img1, img2):
        if self.reshape:
            img1 = img1.view(-1,3,32,32)
            img2 = img2.view(-1,3,32,32)

        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


import cv2
import numpy as np
import matplotlib.pyplot as plt
if __name__=="__main__":
    # torch.set_printoptions(precision=10)
    # np_lena = cv2.imread("lena.jpeg").astype(np.float);
    # np_lena2 = cv2.imread("lena2.jpeg").astype(np.float);
    #
    #
    # lena = torch.from_numpy(np_lena) / 255.0
    # lena_many = torch.stack([lena,lena])
    # lena_many = lena_many.permute(0,3,1,2)
    # lena2 = torch.from_numpy(np_lena) / 255.0
    # lena2_many = torch.stack([lena2,lena2])
    # lena2_many = lena2_many.permute(0,3,1,2)
    # #lena = lena.unsqueeze(0).permute(0,3,1,2)
    # #lena2 = lena.unsqueeze(0).permute(0,3,1,2)
    # #delta = torch.rand(1,3,490,490)
    # #delta = 0.1
    # #delta = delta.to(torch.float64)
    #
    # #lena2 = torch.from_numpy(np_lena) / 255.0
    # #lena2 = lena2.unsqueeze(0).permute(0,3,1,2)
    # #lena = lena+delta
    # # plt.subplot(1,2,1)
    # # plt.imshow(lena2.squeeze().permute(1,2,0).numpy())
    # # plt.subplot(1,2,2)
    # # plt.imshow(lena.squeeze().permute(1,2,0).numpy())
    # #print ( 1 - ssim ( lena , lena2 )) ** 2 / delta;
    # print ssim(lena_many,lena2_many,size_average=False)
    # plt.show()
    ssim_layer = SSIM()
    np_lena = cv2.imread("lena.jpeg").astype(np.float);
    np_lena2 = cv2.imread("lena2.jpeg").astype(np.float);
    lena = torch.from_numpy(np_lena) / 255.0
    lena2 = torch.from_numpy(np_lena) / 255.0
    lena = lena.unsqueeze(0).permute(0,3,1,2)
    lena2 = lena2.unsqueeze(0).permute(0,3,1,2)
    lena = lena.cuda()
    lena2 = lena2.cuda()
    for i in range(10000):
        print i
        ssim_layer(lena,lena2)
