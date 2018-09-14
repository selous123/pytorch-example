import torch.utils.data as data
from PIL import Image
from transform import ReLabel, ToLabel, ToSP, Scale, Colorize
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
import visutils
import torchvision.utils as vutils
import numpy as np
np.set_printoptions(threshold=1e6)
from myfunc import make_image_grid, make_label_grid
import myfunc
import torchvision
import math
import data.dataset as dataset
import model.fcn32model as fcn32model
import model.fcn8stranspose as fcn8smodel
import model.vgg as Vgg
root_directory = "/home/lrh/dataset/VOCdevkit"
model_file = "/home/lrh/.torch/models/vgg16_from_caffe.pth"
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


vocdataset = dataset.VOCDataSet(root_directory,img_transform=input_transform,label_transform=target_transform)
dataloader = data.DataLoader(vocdataset,batch_size=8,drop_last = True)

vgg16_pretrained = torchvision.models.vgg16(pretrained=False)
state_dict = torch.load(model_file)
vgg16_pretrained.load_state_dict(state_dict)

vgg16 = Vgg.Vgg16(vgg16_pretrained).cuda()
Seg = fcn8smodel.Seg().cuda()

optimizer_feat = torch.optim.Adam(vgg16.parameters(), lr=1e-4)
optimizer_seg = torch.optim.Adam(Seg.parameters(), lr=1e-3)


for epoch in range(epoch_num):
    for i,(img,label) in enumerate(dataloader,0):
        # input = make_image_grid(img, mean, std)
        # vutils.save_image(input.cpu().detach(),'%s/img_epoch_%03d.png' % (result_directory,1))
        # break;
        img = img.cuda()
        labels = label[0].cuda()

        feats = vgg16(img)
        logits = Seg(feats)

        vgg16.zero_grad()
        Seg.zero_grad()

        loss = myfunc.cross_entropy2d(logits,labels)
        loss.backward()
        optimizer_seg.step()
        optimizer_feat.step()
        print "epoch is:[{}|{}],index is:[{}|{}],loss:{}".\
                    format(epoch,epoch_num,i,len(dataloader),loss);

    win = visutils.visualize_loss(epoch,loss.cpu().detach(),env,win)

    if epoch%40==0:
        #save model
        torch.save(vgg16.state_dict(),'%s/vgg16_%03d.pkl' %(result_directory,epoch))
        torch.save(Seg.state_dict(),'%s/Seg_%03d.pkl' %(result_directory,epoch))
        #save result

        input = make_image_grid(img, mean, std)
        label = make_label_grid(labels.data)
        label = Colorize()(label).type(torch.FloatTensor)
        output = make_label_grid(torch.max(logits, dim=1)[1].data)
        output = Colorize()(output).type(torch.FloatTensor)

        vutils.save_image(label.cpu().detach(),'%s/labels_epoch_%03d.png' % (result_directory,epoch))
        vutils.save_image(output.cpu().detach(),'%s/output_epoch_%03d.png' % (result_directory,epoch))
        vutils.save_image(input.cpu().detach(),'%s/img_epoch_%03d.png' % (result_directory,epoch))
