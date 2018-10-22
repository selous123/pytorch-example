import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.autograd as autograd
ndf = 128
class feas_model(nn.Module):
    def __init__(self):
        super(feas_model,self).__init__()
        self.feas = nn.Sequential(
            nn.Conv2d(1, ndf, 3, 2, padding=1),
            #nn.LayerNorm([ndf,14,14]),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(),
            nn.Conv2d(ndf, 2 * ndf, 3, 2, padding=1),
            #nn.LayerNorm([2*ndf,7,7]),
            nn.BatchNorm2d(2*ndf),
            nn.LeakyReLU(),
        )

        self.fc = nn.Linear(7*7*2*ndf, 10)

    def forward(self,x):
        #x with shape [batch_size,1,28,28]
        x = self.feas(x)
        #with shape [batch_size,10]
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        return x


model = feas_model()
model = model.cuda()

print model
optimizer = optim.Adam(model.parameters(),lr=1e-4,betas=(0.5,0.999))

data = dset.MNIST(root="/home/lrh/dataset/mnist",train = False,download=True,transform=transforms.Compose([
        transforms.ToTensor(),
        ]))
dataloader = torch.utils.data.DataLoader(data,batch_size=32,shuffle=True,drop_last=True)

for epoch in range(100):
    for i,data in enumerate(dataloader):
        images,label = data
        images = images.cuda()
        images.requires_grad = True
        #image with shape [batch_size,1,28,28]
        feas = model(images)
        gradients = autograd.grad(outputs=feas, inputs=images,
                                  grad_outputs=torch.ones(feas.size()).cuda(),
                                  create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients-1).abs()).mean()
        gradient_penalty.backward()
        optimizer.step()
    print gradients
    print feas
    print "gradient loss is:{}".format(gradient_penalty)
