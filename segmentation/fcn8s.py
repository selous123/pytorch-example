import os
import os.path as osp
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
from myfunc import make_image_grid, make_label_grid
import myfunc
import torchvision
import math
model_file = "/home/lrh/.torch/models/vgg16_from_caffe.pth"

##hyper parameters
root_dir = "/home/lrh/dataset/VOCdevkit";
#["train","val","test"]
split = "train"
epoch_num = 200
env = "segmentation"
win = ""
result_directory = "result_fcn8s_0709"
##loss function
def cross_entropy2d(logits,target,weights=None):
    #logits [b,n_class,h,w] target [b,h,w]
    loss = F.nll_loss(F.log_softmax(logits,dim=1),target,weight=weights)
    #loss_fn = nn.NLLLoss(weight=weights,reduce=False)
    #logit = F.log_softmax(logits,dim=1)
    #print logit.shape
    #print target.shape
    #loss = loss_fn(logit,target)
    #print loss.shape
    #torch.max(logits, dim=1)[1].data
    return loss.unsqueeze(0)
##read data
class VOCDataSet(data.Dataset):
    # class_names = np.array([
    #     'background',
    #     'aeroplane',
    #     'bicycle',
    #     'bird',
    #     'boat',
    #     'bottle',
    #     'bus',
    #     'car',
    #     'cat',
    #     'chair',
    #     'cow',
    #     'diningtable',
    #     'dog',
    #     'horse',
    #     'motorbike',
    #     'person',
    #     'potted plant',
    #     'sheep',
    #     'sofa',
    #     'train',
    #     'tv/monitor',
    # ])
    def __init__(self, root, split="train", img_transform=None, label_transform=None):
        self.root = root
        self.split = split
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.files = []

        data_dir = osp.join(root, "VOC2007")
        imgsets_dir = osp.join(data_dir, "ImageSets/Segmentation/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "JPEGImages/%s.jpg" % name)
                label_file = osp.join(data_dir, "SegmentationClass/%s.png" % name)
                self.files.append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")
        # import numpy as np
        # labels = np.array(label)
        # print np.sum(labels==21)
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

##define network
class FCN8s(nn.Module):
    def __init__(self,n_class = 21):
        super(FCN8s,self).__init__();
        #[3,256,256]
        #conv1
        self.conv1_1 = nn.Conv2d(3,64,3,padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64,64,3,padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        #[64,128,128]
        #conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        #[128,64,64]
        #conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8
        #[256,32,32]<-
        #conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16
        #[512,16,16]<-
        #conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32
        #[512,8,8]
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        #[4096,2,2]

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()


        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)


        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)


    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                print "hello world"
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8


        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        #[512,29,29]

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)


        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        #[4096,9,9]
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        #[4096,9,9]
        h = self.score_fr(h)
        #[21,9,9]

        h = self.upscore2(h)
        upscore2 = h  # 1/16
        #[21,20,20]

        h = self.score_pool4(pool4)
        #[21,29,29]
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        #[21,20,20]
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        #[21,42,42]
        h = self.score_pool3(pool3)
        #[21,57,57]
        #[21,42,42]
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]

        score_pool3c = h  # 1/8
        h = upscore_pool4 + score_pool3c  # 1/8
        h = self.upscore8(h)
        #[21,344,344]
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        #[21,256,256]
        return h


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


class FCN32s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32,
                                          bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.normal_(0)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

# def initialize_weights(nets):
#     for m in nets.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             m.bias.data.normal_(0)
#         if isinstance(m, nn.ConvTranspose2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#             #m.bias.data.normal_(0)
#     # if isinstance(m, nn.ConvTranspose2d):
#     #     assert m.kernel_size[0] == m.kernel_size[1]
#     #     initial_weight = get_upsampling_weight(
#     #         m.in_channels, m.out_channels, m.kernel_size[0])
#     #     m.weight.data.copy_(initial_weight)


vgg = torchvision.models.vgg16(pretrained=False)
state_dict = torch.load(model_file)
vgg.load_state_dict(state_dict)
nets = FCN32s(n_class = 22)
#initialize weights
#initialize_weights(nets)

#nets.copy_params_from_vgg16(vgg)
#state_dict = torch.load(model_file)
#nets.load_state_dict(state_dict)
nets = nets.cuda()

##initialize weights
##test datasets
std = [.229, .224, .225]
mean = [.485, .456, .406]

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
    ReLabel(255, 21),
])

vocdataset = VOCDataSet(root_dir,split,img_transform = input_transform,label_transform=target_transform)
dataloader = data.DataLoader(vocdataset,batch_size=12,drop_last = True)

# weights = torch.ones(22)
# weights[21] = 0
# weights = weights.cuda()

optimizer = torch.optim.Adam(nets.parameters(), lr=1e-3)

import matplotlib.pyplot as plt
for epoch in range(epoch_num):
    for i,data in enumerate(dataloader):
        img,label = data
        img = img.cuda()
        labels = label[0].cuda()
        logits = nets(img)

        output = make_label_grid(torch.max(logits, dim=1)[1].data)
        output = Colorize()(output).type(torch.FloatTensor)
        vutils.save_image(output.cpu().detach(),'%s/output_epoch_%03d.png' % (result_directory,i))

        # print logits.shape
        #print torch.sum(labels>21)
        nets.zero_grad()
        loss = cross_entropy2d(logits,labels)
        loss.backward()
        optimizer.step()
        print "epoch is:[{}|{}],index is:[{}|{}],loss:{}".\
            format(epoch,epoch_num,i,len(dataloader),loss);
    #visualize_loss
    win = visutils.visualize_loss(epoch,loss.cpu().detach(),env,win)

    if epoch%10==0:
        #save model
        torch.save(nets.state_dict(),'%s/net_%03d.pkl' %(result_directory,epoch))
        #save result

        input = make_image_grid(img, mean, std)
        label = make_label_grid(labels.data)
        label = Colorize()(label).type(torch.FloatTensor)
        output = make_label_grid(torch.max(logits, dim=1)[1].data)
        output = Colorize()(output).type(torch.FloatTensor)

        vutils.save_image(label.cpu().detach(),'%s/labels_epoch_%03d.png' % (result_directory,epoch))
        vutils.save_image(output.cpu().detach(),'%s/output_epoch_%03d.png' % (result_directory,epoch))
        vutils.save_image(input.cpu().detach(),'%s/img_epoch_%03d.png' % (result_directory,epoch))
