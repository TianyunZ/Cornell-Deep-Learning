
from torch.nn.modules.activation import LeakyReLU
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

batch_size=50
num_epochs=50
learning_rate=2e-4

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
        x = x.view(-1, 1, 794)
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
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

def noise(n, features_dim=28):
    return Variable(torch.randn(n, features_dim)).to(device)

def noise_labels(n, features_dim=1):
    return Variable(torch.randint(0,10, (n, features_dim))).to(device)

def build_ones(size):
    data = Variable(torch.ones(size, 1))
    return data.to(device)

def build_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)

test_noise = noise(64)

# ======================================= Part 1 ===========================================
generator = Generator(input_dim=38, output_dim=794).to(device)
discriminator = Discriminator(input_dim=794).to(device)

generator.train()
discriminator.train()

g_optim = optim.Adam(generator.parameters(), lr=learning_rate)
d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate)

g_losses = []
d_losses = []
images = []

criterion = nn.BCELoss()

def one_hot(labels):
    pad = torch.zeros(len(labels), class_num)
    for i in range(len(labels)):
        pad[i, labels[i]] = 1
    return pad

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

def train_generator(optimizer, fake_img, labels):
    n = fake_img.size(0)
    optimizer.zero_grad()

    prediction = discriminator(fake_img)
    error = criterion(prediction, build_ones(n))
    
    error.backward()
    optimizer.step()
    
    return error

# To add the conditional input vector, we need to modify both D and G. First, we need to define the input label vector. 
# We are going to use one-hot encoding vectors for labels: for an image sample with label k of K classes, 
# the vector is K dimensional and has 1 at k-th element and 0 otherwise.

# We then concatenate the one-hot encoding of class vector with original image pixels (flattened as a vector) 
# and feed the augmented input to D and G. Note we need to change the number of channels in the first layer accordingly.
class_num = 10
for epoch in range(1, 1+num_epochs):
    g_loss = 0.0
    d_loss = 0.0
    for i, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        temp = imgs.view(batch_size, -1).to(device)
        fake_labels = noise_labels(len(imgs))
        fake_pad = one_hot(fake_labels).to(device)
        fake_input = torch.cat((noise(len(imgs)), fake_pad), 1)
        fake_img = generator(fake_input).detach()
        pad = one_hot(labels).to(device)
        real_input = torch.cat((temp, pad), 1)
        d_loss += train_discriminator(d_optim, real_input, fake_img)

        fake_input = torch.cat((noise(len(imgs)), pad), 1)
        g_loss += train_generator(g_optim, fake_img, labels)

    g_losses.append(g_loss/i)
    d_losses.append(d_loss/i)
    print('Epoch {}: generator_loss: {:.4f} discriminator_loss: {:.4f}\r'.format(epoch, g_loss/i, d_loss/i))

torch.save(generator.state_dict(), 'p5_generator.ckpt')
torch.save(discriminator.state_dict(), 'p5_discriminator.ckpt')
print('Training Finished.')

# ========================== Generate Samples ================================
import numpy as np

generator = Generator(input_dim=38, output_dim=784).to(device)
generator.load_state_dict(torch.load("models/p5_generator.ckpt", map_location=torch.device('cpu')))

to_image = transforms.ToPILImage()
images = [np.array(to_image(i)) for i in images]
np.save("images_list", images)
fake_imgs = []
for i in range(10):
    label = i
    fake_img = generator(noise(3))
    fake_img = generator(noise(len(imgs)))

