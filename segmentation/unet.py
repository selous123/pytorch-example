# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from data.dataset import VOCDataSet
import torch.utils.data as data
from transform import ReLabel, ToLabel, ToSP, Scale, Colorize
import torchvision.transforms as transforms
from PIL import Image
import model.unetmodel as UNet
root_directory = "/home/lrh/dataset/VOCdevkit"
std = [.229, .224, .225]
mean = [.485, .456, .406]
result_directory = "result_fcn8s_0709"
epoch_num = 500
env = "segmentation"
win = ""

input_transform = transforms.Compose([
    Scale((256, 256), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
target_transform = transforms.Compose([
    Scale((256, 256), Image.NEAREST),
    ToSP(256),
    #transforms.ToTensor(),
    ToLabel(),
    ReLabel(255,0),
])

n_classes = 22
dataset = VOCDataSet(root_directory,img_transform=input_transform,label_transform=target_transform)
dataloader = data.DataLoader(dataset,batch_size=8,shuffle=True,drop_last=False)



net = UNet(3,n_classes)

print net
# weights = torch.ones(22)
# weights[21] = 0
# weights = weights.cuda()
#
# optimizer = torch.optim.Adam(nets.parameters(), lr=1e-3)
optimizer = optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=0.9,
                      weight_decay=0.0005)
# for epoch in range(epoch_num)
#     for i,data in enumerate(dataloader):
#         images,labels = data
#         logits = net(images)
#
#
#         net.zero_grad()
#         loss = cross_entropy2d(logits,labels,weights)
#         loss.backward()
#         optimizer.step();
#         print "epoch is:[{}|{}],index is:[{}|{}],loss:{}".\
#             format(epoch,epoch_num,i,len(dataloader),loss);
#     if epoch%10==0:
#         #save model
#         torch.save(net.state_dict(),"%s/unet_%03d.pkl" %(result_directory,epoch));
