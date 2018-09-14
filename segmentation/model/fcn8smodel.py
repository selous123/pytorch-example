import torch.nn as nn
import math
class Seg(nn.Module):
    def __init__(self):
        super(Seg,self).__init__()
        #with shape [[256,32,32],[512,16,16],[512,8*8]]
        #->shape [256,256]
        self.conv = nn.ModuleList([
            nn.Conv2d(256,21,kernel_size=1),
            nn.Conv2d(512,21,kernel_size=1),
            nn.Conv2d(4096,21,kernel_size=1)
        ])
        self.deconv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(21,21,kernel_size=1),
                nn.Upsample(scale_factor=8)
            ),
            nn.Sequential(
                nn.Conv2d(21,21,kernel_size=1),
                nn.Upsample(scale_factor=2)
            )
        ])

        self.deconv2 = nn.Sequential(
            nn.Conv2d(21,21,kernel_size=1),
            nn.Upsample(scale_factor=8)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.normal_(0)


    def forward(self,feats):
        #feats:List with shape with shape [[256,32,32],[512,16,16],[4096,2*2]]
        #->feats:List with shape with shape [[21,32,32],[21,16,16],[21,2*2]]
        for i in range(3):
            feats[i] = self.conv[i](feats[i])
        i = 1
        x = feats[2]
        for l in self.deconv:
            x = l(x)
            x = x+feats[i]
            i = i-1;
        #[21,32,32]
        x = self.deconv2(x)
        #[21,256,256]
        return x
if __name__=="__main__":
    net = Seg()
    print net
