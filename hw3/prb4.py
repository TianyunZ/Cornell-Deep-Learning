import torchvision
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.optim as optim
from torchvision.utils import make_grid
import copy
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size=50
num_epochs=50
learning_rate=2e-4

# fmnist.data => torch.Size([60000, 28, 28])
fmnist = datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=fmnist, batch_size=batch_size, shuffle=True)


class Generator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Sequential(
                    nn.Linear(self.input_dim, 256),
                    nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU()
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(512, self.output_dim),
                    nn.Tanh()
                    )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view(-1, 1, self.input_dim, self.input_dim)
        return x
 
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.out_dim = 1
        self.fc1 = nn.Sequential(
                    nn.Linear(self.input_dim, 512),
                    nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU()
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.out_dim),
                    nn.Sigmoid() # If WGAN, comment
                    )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def noise(n, features_dim=28):
    return Variable(torch.randn(n, features_dim)).to(device)

def build_ones(size):
    data = Variable(torch.ones(size, 1))
    return data.to(device)

def build_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)

test_noise = noise(64)

generator = Generator(input_dim=28, output_dim=784).to(device)
discriminator = Discriminator(input_dim=784).to(device)

generator.train()
discriminator.train()

g_optim = optim.Adam(generator.parameters(), lr=learning_rate)
d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate)

g_losses = []
d_losses = []
images = []

criterion = nn.BCELoss()

def train_discriminator(optimizer, real_img, fake_img):
    n = real_img.size(0)

    optimizer.zero_grad()
    
    pred_real = discriminator(real_img)
    error_real = criterion(pred_real, build_ones(n))
    error_real.backward()

    pred_fake = discriminator(fake_img)
    error_fake = criterion(pred_fake, build_zeros(n))
    
    error_fake.backward()
    optimizer.step()
    
    return error_real + error_fake

def train_generator(unroll_D, optimizer, fake_img):
    n = fake_img.size(0)
    optimizer.zero_grad()
    
    prediction = unroll_D(fake_img)
    error = criterion(prediction, build_ones(n))
    
    error.backward()
    optimizer.step()
    
    return error

def train_unroll_D(unroll_D, optimizer, real_img, fake_img):
    n = real_img.size(0)

    optimizer.zero_grad()
    
    pred_real = unroll_D(real_img)
    error_real = criterion(pred_real, build_ones(n))
    error_real.backward()

    pred_fake = unroll_D(fake_img)
    error_fake = criterion(pred_fake, build_zeros(n))
    
    error_fake.backward()
    optimizer.step()
    
    return error_real + error_fake


k = 3
for epoch in range(1, 1+num_epochs):
    g_loss = 0.0
    d_loss = 0.0
    un_d_loss = 0.0
    for i, data in enumerate(data_loader):
        imgs, _ = data
        fake_img = generator(noise(len(imgs))).detach()
        real_img = imgs.to(device)
        d_loss += train_discriminator(d_optim, real_img, fake_img)

        # Make a copy of D into D_unroll
        torch.save(discriminator.state_dict(), "temp.ckpt")
        unroll_D = Discriminator(input_dim=784).to(device)
        unroll_D.load_state_dict(torch.load("temp.ckpt"))
        for _ in range(k):
            imgs, _ = data
            fake_img = generator(noise(len(imgs))).detach()
            real_img = imgs.to(device)
            un_d_loss += train_unroll_D(unroll_D, d_optim, real_img, fake_img)

        fake_img = generator(noise(len(imgs)))
        g_loss += train_generator(unroll_D, g_optim, fake_img)

    img = generator(test_noise).cpu().detach()
    img = make_grid(img)
    images.append(img)
    g_losses.append(g_loss/i)
    d_losses.append(d_loss/i)
    print('Epoch {}: generator_loss: {:.4f} discriminator_loss: {:.4f}\r'.format(epoch, g_loss/i, d_loss/i))
    
print('Training Finished.')

# ==============================================================================
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.out_dim = 10
        self.fc1 = nn.Sequential(
                    nn.Linear(self.input_dim, 512),
                    nn.ReLU()
                    )
        self.fc2 = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU()
                    )
        self.fc3 = nn.Sequential(
                    nn.Linear(256, self.out_dim),
                    # nn.Sigmoid() # If WGAN, comment
                    nn.Softmax()
                    )

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

discriminator = Discriminator(input_dim=784)
discriminator.load_state_dict(torch.load("models/p3_discriminator.ckpt", map_location=torch.device('cpu')))
# discriminator.load_state_dict(torch.load("models/wgan/wgan_discriminator.ckpt", map_location=torch.device('cpu')))

generator = Generator(input_dim=28, output_dim=784).to(device)
generator.load_state_dict(torch.load("models/wgan/wgan_generator.ckpt", map_location=torch.device('cpu')))
# generator.load_state_dict(torch.load("models/p4_unroll_gan.ckpt", map_location=torch.device('cpu')))
test_noise = noise(1)

# images = []
# for i in range(3000):
#     img = generator(test_noise).cpu().detach()
#     img = make_grid(img)
#     images.append(img)

import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from PIL import Image

test_noise = noise(64)
img = generator(test_noise).cpu().detach()
# img = real_img
img = make_grid(img)
to_image = transforms.ToPILImage()
a = np.array(to_image(img))
print(a.shape)
im = Image.fromarray(a)
plt.imshow(im)
im.save("temp.jpg")
print('finish')
