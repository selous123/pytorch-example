import torch.nn as nn
import math
import torch
import numpy as np

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class Seg(nn.Module):
    def __init__(self):
        super(Seg,self).__init__()
        #with shape [[256,32,32],[512,16,16],[512,8*8]]
        #->shape [256,256]
        n_class = 21
        self.conv = nn.ModuleList([
            nn.Conv2d(256,21,kernel_size=1),
            nn.Conv2d(512,21,kernel_size=1),
            nn.Conv2d(4096,21,kernel_size=1)
        ])


        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)



        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.normal_(0)

        if isinstance(m, nn.ConvTranspose2d):
            assert m.kernel_size[0] == m.kernel_size[1]
            initial_weight = get_upsampling_weight(
                m.in_channels, m.out_channels, m.kernel_size[0])
            m.weight.data.copy_(initial_weight)


    def forward(self,feats):
        #feats:List with shape with shape [[256,57,57],[512,29,29],[4096,9*9]]
        #->feats:List with shape with shape [[21,57,57],[21,29,29],[21,9*9]]
        for i in range(3):
            feats[i] = self.conv[i](feats[i])

        x = feats[2]
        x = self.upscore2(x)
        upscore2 = x  # 1/16
        #[21,20,20]

        x = feats[1]
        #[21,29,29]
        x = x[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = x  # 1/16

        x = upscore2 + score_pool4c  # 1/16
        #[21,20,20]
        x = self.upscore_pool4(x)
        upscore_pool4 = x  # 1/8
        #[21,42,42]
        x = feats[0]
        #[21,57,57]
        #[21,42,42]
        x = x[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]

        score_pool3c = x  # 1/8
        x = upscore_pool4 + score_pool3c  # 1/8
        x = self.upscore8(x)
        #[21,344,344]
        x = x[:, :, 31:31 + 256, 31:31 + 256].contiguous()
        #[21,256,256]
        return x
if __name__=="__main__":
    net = Seg()
    print net
