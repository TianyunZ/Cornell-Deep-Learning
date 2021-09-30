import torch
import sys
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from PIL import Image
from torch.autograd import Variable
import time
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd

total_class = 37

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()
# print(model.eval())
# sys.exit()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# for param in model.parameters():
#     param.requires_grad = False

fc_inputs = model.fc.in_features
# print(fc_inputs)
model.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 37),
    nn.LogSoftmax(dim=1)
)

dataset = 'oxford_pet_37'
train_directory = os.path.join(dataset, 'train')
test_directory = os.path.join(dataset, 'test')

batch_size = 16
num_classes = 37
 
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=preprocess),
    'test': datasets.ImageFolder(root=test_directory, transform=preprocess)
 
}

train_data_size = len(data['train'])
test_data_size = len(data['test'])
print("train_data_size:",train_data_size)
print("test_data_size:",test_data_size)
 
train_data = DataLoader(data['train'], batch_size=50, shuffle=False)
test_data = DataLoader(data['test'], batch_size=50, shuffle=False)

class TransferModel(nn.Module):
    def __init__(self, input_dim, out_dim, num_epochs, batch_size):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.loss_function = nn.CrossEntropyLoss()
        self.input_dim = input_dim
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.2)
        print(self.input_dim, self.batch_size, self.input_dim*self.batch_size)
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, out_dim)
        self.optimizer = torch.optim.Adam(params = self.parameters(), lr=1e-4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def train(self, training_data, test_data):
        criterion = self.loss_function
        for epoch in range(0, self.epochs):

            running_loss = 0
            round = 0
            
            train_loader = torch.utils.data.DataLoader(training_data, batch_size=self.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

            for data, label in train_loader:
                data = torch.FloatTensor(data.float()).to(self.device)
                label = torch.LongTensor(label.long()).to(self.device)
                label = label.view(label.size(0))
                out = self.forward(data)

                loss = criterion(out, label)
                running_loss += loss.data
                round += 1
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                    
            test_acc = 0.0
            for data, label in test_loader:
                data = Variable(torch.FloatTensor(data.float())).to(self.device)
                outputs = self.forward(data)
 
                _, pred = torch.max(outputs.data, 1)
                correct_counts = pred.eq(label.data.view_as(pred))
 
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
 
                test_acc += acc.item() * data.size(0)

            avg_test_acc = test_acc/test_data_size
            print(avg_test_acc*100)

model_weights = []
conv_layers = []

model_children = list(model.children())

counter = 0 
dim = 512

train_data = DataLoader(data['train'], batch_size=50, shuffle=False)
test_data = DataLoader(data['test'], batch_size=50, shuffle=False)

train_data_list = []
train_label_list = []
test_data_list = []
test_label_list = []

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)

#  pre-train model with model_children
for idx, (inputs, labels) in tqdm(enumerate(test_data), "Pre-training test data..."):
    test_label_list.append(labels.tolist())
    outputs = [model_children[0](inputs)]
    for i in range(1, len(model_children)-1):
        outputs.append(model_children[i](outputs[-1]))
    feature = outputs[len(outputs)-1][:, :, :, :]
    feature = feature.data # torch.Size([512, 14, 14])

    feature = feature.view(-1,dim)
    test_data_list.append(feature.tolist())
np.concatenate(test_data_list)
np.save('pre-data/test/data.npy', test_data_list)
np.save('pre-data/test/label.npy', test_label_list)

for idx, (inputs, labels) in tqdm(enumerate(train_data), "Pre-training train data..."):
    train_label_list.append(labels.tolist())
    outputs = [model_children[0](inputs)]
    for i in range(1, len(model_children)-1):
        outputs.append(model_children[i](outputs[-1]))
    feature = outputs[len(outputs)-1][:, :, :, :]
    feature = feature.data # torch.Size([512, 14, 14])

    feature = feature.view(-1,dim)
    train_data_list.append(feature.tolist())
np.concatenate(train_data_list)
np.save('pre-data/train/data.npy', train_data_list)
np.save('pre-data/train/label.npy', train_label_list)

norm = []
train_data_list = np.load("pre-data/train/data.npy",allow_pickle=True)
train_data_list = np.concatenate(train_data_list, axis=0)
norm.append(np.sqrt(np.sum(train_data_list*train_data_list, axis=0)))
np.save("pre-data/norm_sub.npy", norm)

train_data_list = train_data_list/norm
np.save('pre-data/train/norm.npy', train_data_list)

test_data_list = np.load('pre-data/test/data.npy',allow_pickle=True)
test_data_list = np.concatenate(test_data_list, axis=0)
test_data_list = test_data_list/norm
np.save('pre-data/test/norm.npy', test_data_list)

train_label_list = np.load("pre-data/train/label.npy",allow_pickle=True)
train_label_list = np.concatenate(train_label_list)
print(train_label_list, len(train_data_list))
print(len(train_data_list))
test_label_list = np.load("pre-data/test/label.npy",allow_pickle=True)
test_label_list = np.concatenate(test_label_list)
print(test_label_list, len(test_label_list))
print(len(test_data_list))


PATH = "models/problem2.pt"
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))
else:
    model = TransferModel(dim, total_class, 330, batch_size)

    model.train(list(zip(train_data_list, train_label_list)), list(zip(test_data_list, test_label_list)))


# for idx, (inputs, labels) in tqdm(enumerate(test_data), "Pre-training test data..."):
#     if idx >= 20:
#         break
#     test_label_list.append(labels.tolist())
#     outputs = [conv_layers[0](inputs)]
#     for i in range(1, len(conv_layers)):
#         outputs.append(conv_layers[i](outputs[-1]))

#     feature = outputs[len(outputs)-1][:, :, :, :]
#     feature = feature.data # torch.Size([512, 14, 14])
#     # normed_layer = F.normalize(feature)
#     feature = feature.view(-1,dim)
#     test_data_list.append(feature.tolist())
# np.concatenate(test_data_list)
# np.save('pre-data/test/batch_1.npy', test_data_list)
# df = pd.DataFrame(test_data_list)
# df.to_csv('pre-data/test_data.csv')
# df = pd.DataFrame(test_label_list)
# df.to_csv('pre-data/test_label.csv')

# for idx, (inputs, labels) in tqdm(enumerate(test_data), "Pre-training test data..."):
#     if idx < 20:
#         continue
#     test_label_list.append(labels.tolist())
#     outputs = [conv_layers[0](inputs)]
#     for i in range(1, len(conv_layers)):
#         outputs.append(conv_layers[i](outputs[-1]))

#     feature = outputs[len(outputs)-1][:, :, :, :]
#     feature = feature.data # torch.Size([512, 14, 14])
#     # normed_layer = F.normalize(feature)
#     feature = feature.view(-1, dim)
#     test_data_list.append(feature.tolist())
# np.save('pre-data/test/batch_2.npy', test_data_list)
# df = pd.DataFrame(test_data_list)
# df.to_csv('pre-data/test_data.csv', mode='a')
# df = pd.DataFrame(test_label_list)
# df.to_csv('pre-data/test_label.csv', mode='a')
# np.save('temp.npy',test_data_list)

# train_label_list = []
# for idx, (inputs, labels) in tqdm(enumerate(train_data), "Pre-training train data..."):
#     if idx >= 20:
#         break
#     train_label_list.append(labels.tolist())
#     outputs = [conv_layers[0](inputs)]
#     for i in range(1, len(conv_layers)):
#         outputs.append(conv_layers[i](outputs[-1]))

#     feature = outputs[len(outputs)-1][:, :, :, :]
#     feature = feature.data # torch.Size([512, 14, 14])
#     # print(feature.size())
#     feature = feature.view(-1, dim)
#     # normed_layer = F.normalize(feature)
#     train_data_list.append(feature.tolist())
# np.save('pre-data/train/batch_0.npy', train_data_list)
#     # print(feature.tolist())
#     # if dim == 1:
#     #     for d in feature.size():
#     #         dim *= d
#         # print("dim:", dim)
# df = pd.DataFrame(train_data_list)
# df.to_csv('pre-data/train_data.csv')
# df = pd.DataFrame(train_label_list)
# df.to_csv('pre-data/train_label.csv')

# flg = 0
# while (flg <= 80):
#     flg += 20
#     train_data_list = []
#     for idx, (inputs, labels) in tqdm(enumerate(train_data), "Pre-training train data..."):
#         if idx < flg:
#             continue
#         if idx >= flg + 20:
#             break
#         train_label_list.append(labels.tolist())
#         outputs = [conv_layers[0](inputs)]
#         for i in range(1, len(conv_layers)):
#             outputs.append(conv_layers[i](outputs[-1]))

#         feature = outputs[len(outputs)-1][:, :, :, :]
#         feature = feature.data # torch.Size([512, 14, 14])
#         # print(feature.size())
#         feature = feature.view(-1, dim)
#         # normed_layer = F.normalize(feature)
#         train_data_list.append(feature.tolist())
    # np.save('pre-data/train/batch_'+str(int(flg/20))+'.npy', train_data_list)
    # sys.exit()
#     df = pd.DataFrame(train_data_list)
#     df.to_csv('pre-data/train_data.csv', mode='a')
#     df = pd.DataFrame(train_label_list)
#     df.to_csv('pre-data/train_label.csv', mode='a')


# np.save('pre-data/train_data', train_data_list)
# np.save('pre-data/train_label', train_label_list)
# train_data = DataLoader(zip(train_data_list, train_label_list), batch_size=batch_size, shuffle=True)


# np.save('pre-data/test_data', test_data_list)
# np.save('pre-data/test_label', test_label_list)
# # test_data = DataLoader(zip(test_data_list, test_label_list), batch_size=batch_size, shuffle=True)
# k = 0
# norm = []
# norm_final = []
# while k<dim:
#     temp = []
#     print(k/4000)
#     for i in range(6):
#         train_data_list = np.load('pre-data/train/batch_'+str(i)+'.npy',allow_pickle=True)
#         a = np.concatenate(train_data_list, axis=0)
#         # print(a.shape)
#         temp.append(np.concatenate(train_data_list)[:,k:k+4000])
#     a = np.concatenate(temp)
#     norm.append(np.sqrt(np.sum(a*a, axis=0)))
#     k += 4000
# norm_final = np.concatenate(norm)
# np.save("pre-data/train_norm/norm.npy", norm_final)
# print(norm_final)
# print(norm_final.shape)

# for i in range(6):
#     train_data_list = np.load('pre-data/train/batch_'+str(i)+'.npy',allow_pickle=True)
#     a = np.concatenate(train_data_list, axis=0)
#     b = a/norm_final
#     np.save('pre-data/train_norm/batch_'+str(i)+'.npy', b)

# for i in range(1,3):
#     data_list = np.load('pre-data/test/batch_'+str(i)+'.npy',allow_pickle=True)
#     a = np.concatenate(data_list, axis=0)
#     b = a/norm_final
#     np.save('pre-data/test_norm/batch_'+str(int(i-1))+'.npy', b)

# sys.exit()

# train_data_list = torch.Tensor(temp)
# print(train_data_list)
# print(train_data_list.shape)
# train_label_list = np.load("pre-data/train_label.npy", allow_pickle=True)
# test_data_list = np.load("pre-data/test_data.npy", allow_pickle=True)
# test_label_list = np.load("pre-data/test_label.npy", allow_pickle=True)

# df = pd.read_csv("pre-data/test_data.csv")
# data = df.values
# norm = np.sqrt(np.sum(data*data, axis=0))
# data_list = data/norm
# df = pd.DataFrame(data_list)
# df.to_csv('pre-data/normed_test_data.csv')

# df = pd.read_csv("pre-data/train_data.csv")
# data = df.values
# norm = np.sqrt(np.sum(data*data, axis=0))
# data_list = data/norm
# # print(data_list)
# df = pd.DataFrame(data_list)
# df.to_csv('pre-data/normed_train_data.csv')


# power_list = [0]*dim
# for index, row in train_data_list.iteritems():
#     print(row)
#     # for i in range(row.size()):
#     #     power_list[i] += row[i]*row[i]


# print(train_label_list)
# print(type(train_label_list))


# train_data = DataLoader(data['train'], batch_size=50, shuffle=True)
# test_data = DataLoader(data['test'], batch_size=50, shuffle=True)

