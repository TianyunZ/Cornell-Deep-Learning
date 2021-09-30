import torchvision
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.optim as optim
from torchvision.utils import make_grid

transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size=96
num_epochs=50
learning_rate=1e-4

# fmnist.data => torch.Size([60000, 28, 28])
fmnist = datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=fmnist, batch_size=batch_size, shuffle=True)

# Generator model
# Make your generator to be a simple network with three linear hidden layers with ReLU activation functions. 
# For the output layer activation function, you should use hyperbolic tangent (tanh). 
# This is typically used as the output for the generator because ReLU cannot output negative values. 
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

# Discriminator
# Make your discriminator to be a similar network with three linear hidden layers using ReLU activation functions, 
# but the last layer should have a logistic sigmoid as its output activation function, 
# since it the discriminator D predicts a score between 0 and 1, where 0 means fake and 1 means real. 
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

# # ======================================= Part 1 ===========================================
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

def train_generator(optimizer, fake_img):
    n = fake_img.size(0)
    optimizer.zero_grad()
    
    prediction = discriminator(fake_img)
    error = criterion(prediction, build_ones(n))
    
    error.backward()
    optimizer.step()
    
    return error

for epoch in range(1, 1+num_epochs):
    g_loss = 0.0
    d_loss = 0.0
    for i, data in enumerate(data_loader):
        imgs, _ = data
        fake_img = generator(noise(len(imgs))).detach()
        real_img = imgs.to(device)
        d_loss += train_discriminator(d_optim, real_img, fake_img)

        fake_img = generator(noise(len(imgs)))
        g_loss += train_generator(g_optim, fake_img)

    img = generator(test_noise).cpu().detach()
    img = make_grid(img)
    images.append(img)
    g_losses.append(g_loss/i)
    d_losses.append(d_loss/i)
    print('Epoch {}: generator_loss: {:.4f} discriminator_loss: {:.4f}\r'.format(epoch, g_loss/i, d_loss/i))
    
print('Training Finished.')

# # ==========================================================================================

# ======================================= Part 2 - MSE ===========================================
generator = Generator(input_dim=28, output_dim=784).to(device)
discriminator = Discriminator(input_dim=784).to(device)
generator.train()
g_optim = optim.Adam(generator.parameters(), lr=learning_rate)

g_losses = []
criterion = nn.MSELoss()

def train_generator_MSE(optimizer, real_img, fake_img):
    n = real_img.size(0)

    optimizer.zero_grad()
    
    error = criterion(fake_img, real_img)
    error.backward()

    optimizer.step()
    
    return error

for epoch in range(1, 1+num_epochs):
    g_loss = 0.0
    for i, data in enumerate(data_loader):
        imgs, _ = data
        fake_img = generator(noise(len(imgs)))
        real_img = imgs.to(device)
        g_loss += train_generator_MSE(g_optim, real_img, fake_img)
    g_losses.append(g_loss/i)
    print('Epoch {}: generator_loss: {:.4f}\r'.format(epoch, g_loss/i))
    
print('Training Finished.')
# ==========================================================================================

# ======================================= Part 2 - WGAN ===========================================
generator = Generator(input_dim=28, output_dim=784).to(device)
discriminator = Discriminator(input_dim=784).to(device)
generator.train()
discriminator.train()

g_losses = []
d_losses = []
criterion = nn.L1Loss()

c = [0.1, 0.01, 0.001, 0.0001]
weight_clip_limit = c[1]

# WGAN with gradient clipping uses RMSprop instead of ADAM
d_optim = torch.optim.RMSprop(discriminator.parameters(), lr=learning_rate)
g_optim = torch.optim.RMSprop(generator.parameters(), lr=learning_rate)

def train_discriminator_WGAN(optimizer, real_img, fake_img):
    n = real_img.size(0)

    optimizer.zero_grad()
    
    pred_real = discriminator(real_img)
    # error_real = -1*criterion(pred_real, build_zeros(n))
    error_real = 1 * pred_real.mean(0).view(1)
    error_real.backward()

    pred_fake = discriminator(fake_img)
    # error_fake = criterion(pred_fake, build_zeros(n))
    error_fake = -1 * pred_fake.mean(0).view(1)
    
    error_fake.backward()
    optimizer.step()
    
    return error_real + error_fake

def train_generator_WGAN(optimizer, fake_img):
    n = fake_img.size(0)
    optimizer.zero_grad()
    
    prediction = discriminator(fake_img)
    # error = -1 * criterion(prediction, build_zeros(n))
    error = 1 * prediction.mean(0).view(1)
    
    error.backward()
    optimizer.step()
    
    return error

for epoch in range(1, 1+num_epochs):
    d_loss = 0.0
    g_loss = 0.0
    for i, data in enumerate(data_loader):
        imgs, _ = data
        for p in discriminator.parameters():
            p.data.clamp_(min=-weight_clip_limit, max=weight_clip_limit)
        fake_img = generator(noise(len(imgs))).detach()
        real_img = imgs.to(device)
        d_loss += train_discriminator_WGAN(d_optim, real_img, fake_img)

        fake_img = generator(noise(len(imgs)))
        g_loss += train_generator_WGAN(g_optim, fake_img)
    d_losses.append(d_loss/i)
    g_losses.append(g_loss/i)
    print('Epoch {}: generator_loss: {:.4f} discriminator_loss: {:.4f}\r'.format(epoch, g_loss.item()/i, d_loss.item()/i))
    
print('Training Finished.')
# ==========================================================================================


# ======================================= Part 2 - LSGAN ===========================================
generator = Generator(input_dim=28, output_dim=784).to(device)
discriminator = Discriminator(input_dim=784).to(device)
generator.train()
discriminator.train()

g_optim = optim.Adam(generator.parameters(), lr=learning_rate)
d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate)

g_losses = []
d_losses = []
criterion = nn.MSELoss()

def train_discriminator_LSGAN(optimizer, real_img, fake_img):
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

def train_generator_LSGAN(optimizer, fake_img):
    n = fake_img.size(0)
    optimizer.zero_grad()
    
    prediction = discriminator(fake_img)
    error = criterion(prediction, build_ones(n))
    
    error.backward()
    optimizer.step()
    
    return error

for epoch in range(1, 1+num_epochs):
    d_loss = 0.0
    g_loss = 0.0
    for i, data in enumerate(data_loader):
        imgs, _ = data
        for p in discriminator.parameters():
            p.data.clamp_(max=c[0])
        fake_img = generator(noise(len(imgs))).detach()
        real_img = imgs.to(device)
        d_loss += train_discriminator_LSGAN(d_optim, real_img, fake_img)

        fake_img = generator(noise(len(imgs)))
        g_loss += train_generator_LSGAN(g_optim, fake_img)
    d_losses.append(d_loss/i)
    g_losses.append(g_loss/i)
    print('Epoch {}: generator_loss: {:.4f} discriminator_loss: {:.4f}\r'.format(epoch, g_loss/i, d_loss/i))
    
print('Training Finished.')
# ==========================================================================================

import numpy as np
from matplotlib import pyplot as plt
import imageio
to_image = transforms.ToPILImage()
# images = [np.array(to_image(i)) for i in images]
# np.save("images_list", images)
# imageio.mimsave('progress.gif', imgs[0])

plt.plot(g_losses.detach().numpy(), label='Generator_Losses')
plt.plot(d_losses.detach().numpy(), label='Discriminator Losses')
plt.legend()
plt.savefig('loss.png')
