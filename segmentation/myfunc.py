from torchvision.utils import make_grid
import torch.nn.functional as F

def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img

def make_label_grid(label):
    label = make_grid(label.unsqueeze(1).expand(-1, 3, -1, -1))[0:1]
    return label

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
