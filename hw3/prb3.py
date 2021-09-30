import torchvision
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
import torch.optim as optim
from torchvision.utils import make_grid
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
train_fmnist = datasets.FashionMNIST(root="./", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_fmnist, batch_size=batch_size, shuffle=True)

test_fmnist = datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_fmnist, batch_size=batch_size, shuffle=True)

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

def noise(n, features_dim=28):
    return Variable(torch.randn(n, features_dim)).to(device)

def build_ones(size):
    data = Variable(torch.ones(size, 1))
    return data.to(device)

def build_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data.to(device)

test_noise = noise(64)


discriminator = Discriminator(input_dim=784).to(device)

discriminator.train()

optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

g_losses = []
d_losses = []
images = []

criterion = nn.CrossEntropyLoss()

def train_discriminator(optimizer, x, y):
    optimizer.zero_grad()
    
    pred = discriminator(x)
    error = criterion(pred, y)
    error.backward()

    optimizer.step()
    
    return error


train_losses = []
test_acc = []

for epoch in range(1, num_epochs + 1):
    batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.view([batch_size, -1]).to(device)
        y_batch = y_batch.to(device)
        loss = train_discriminator(optimizer, x_batch, y_batch)
        batch_losses.append(loss.item())
    training_loss = np.mean(batch_losses)
    train_losses.append(training_loss)

    with torch.no_grad():
        batch_test_acc = []
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.view([batch_size, -1]).to(device)
            y_batch = y_batch.to(device)
            outputs = discriminator(x_batch)

            _, pred = torch.max(outputs.data, 1)
            correct_counts = pred.eq(y_batch.data.view_as(pred))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            batch_test_acc.append(acc)

        avg_test_acc = np.mean(batch_test_acc)
        test_acc.append(avg_test_acc)

    print(
        f"[{epoch}/{num_epochs}] Training loss: {training_loss:.4f}\t Test Acc: {avg_test_acc:.4f}"
    )
    
print('Training Finished.')

test_fmnist = datasets.FashionMNIST(root="./", train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_fmnist, batch_size=3000, shuffle=True)

# discriminator = torch.load("models/p3_discriminator.ckpt", map_location=torch.device('cpu'))
discriminator = Discriminator(input_dim=784)
discriminator.load_state_dict(torch.load("models/p3_discriminator.ckpt", map_location=torch.device('cpu')))
for x_batch, y_batch in test_loader:
    x_batch = x_batch.view([3000, -1]).to(device)
    y_batch = y_batch.to(device)
    outputs = discriminator(x_batch)
    _, pred = torch.max(outputs.data, 1)
    print(pred)
    print(y_batch)
    break

import matplotlib.pyplot as plt
import numpy as np

pred = pred.tolist()
y_batch = y_batch.tolist()
plt.hist(pred, edgecolor='k', alpha=0.35, bins=10)
plt.title("Prediction Distribution of 3000 Samples")
plt.xlabel("Prediction")
plt.ylabel("Numbers")
plt.show()

# plt.hist(y_batch, edgecolor='k', alpha=0.35, bins=10)
# plt.title("Labels Distribution of 3000 Samples")
# plt.xlabel("Label")
# plt.ylabel("Numbers")
# plt.show()
