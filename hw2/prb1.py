import torch
import sys
import numpy as np
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

# Part 1
from PIL import Image
from torchvision import transforms

filename = "peppers.jpg"
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
org = Image.open(filename)
input_tensor = preprocess(org)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output.size())
# print(output[0].size())
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities.size())

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top3_prob, top3_catid = torch.topk(probabilities, 3)
for i in range(top3_prob.size(0)):
    print(categories[top3_catid[i]], top3_prob[i].item())
# sys.exit()

# Part 2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
# import cv2 as cv

model_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # we will save the 49 conv layers in this list

# get all the model children as list
model_children = list(model.children())
# print(model)

# sys.exit()

# counter to keep count of the conv layers
counter = 0 
# append all the conv layers and their respective weights to the list
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
print(f"Total convolutional layers: {counter}")


# # visualize the first conv layer filters
# plt.figure(figsize=(20, 17))
# for i, filter in enumerate(model_weights[0]):
#     plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
#     plt.imshow(filter[0, :, :].detach(), cmap='gray')
#     plt.axis('off')
#     plt.savefig('outputs/filter.png')
# plt.show()

print(input_tensor.size())
print(input_batch.size())

# pass the image through all the layers
outputs = [conv_layers[0](input_batch)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    outputs.append(conv_layers[i](outputs[-1]))

# visualize 64 features from each layer
for idx in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_data = outputs[idx][0, :, :, :]
    layer_data = layer_data.data
    print(layer_data.size()) # [64, 112, 112]
    for i, filter in enumerate(layer_data):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {idx} feature maps...")
    plt.savefig(f"outputs/layer_{idx}.png")
    # plt.show()
    plt.close()