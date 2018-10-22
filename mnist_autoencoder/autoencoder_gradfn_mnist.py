import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import torch.autograd as autograd
if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 200
batch_size = 128
learning_rate = 1e-3
LAMBDA = 10

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('/home/lrh/dataset/mnist', train = True,transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.fc = nn.Linear(8*2*2,10)
    def forward(self,x):
        x = self.main(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        return x
class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()

        self.fc = nn.Linear(10,8*2*2)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self,x):
        x = self.fc(x)
        x = x.reshape(x.size(0),8,2,2)
        x = self.main(x)
        return x




encoder = encoder().cuda()
decoder = decoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=learning_rate,
                             weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.cuda()
        img.requires_grad = True
        optimizer.zero_grad()

        feas = encoder(img)
        output = decoder(feas)
        mse = criterion(output, img.detach())
        mse.backward(retain_graph=True)
        gradients = autograd.grad(outputs=feas, inputs=img,
                                  grad_outputs=torch.ones(feas.size()).cuda(),
                                  create_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        #print gradients
        #print "feas:{}".format(feas)
        gradient_penalty = ((gradients-1) ** 2).mean()*LAMBDA
        gradient_penalty.backward()

        optimizer.step()



    # ===================log========================
    print gradients.detach().cpu().numpy()[0]
    #print "feas:{}".format(feas)
    print('epoch [{}/{}], loss:{:.4f},gradient_penalty:{:.4f}'
          .format(epoch+1, num_epochs, mse,gradient_penalty))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './edfngrad/image_{}.png'.format(epoch))

torch.save(encoder.state_dict(), './edfngrad/encoder.pth')
