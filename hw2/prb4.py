
import os
import cv2
import copy
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import models
from torch.nn import functional
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image

class FoolingCNNModel():

    def __init__(self, model, init_img, org_label, target_label, min_confidence):
        self.model = model
        self.model.eval()
        self.target_label = target_label
        self.min_confidence = min_confidence
        self.init_img = init_img
        self.old_img = None
        self.org_label = org_label
        self.clip = 5

        if not os.path.exists('generated'):
            os.makedirs('generated')

    def generate(self):
        # criterion = nn.CrossEntropyLoss()
        for i in range(1, 200):

            self.processed_img = preprocess_image(self.init_img)
            self.old_img = self.processed_img

            optimizer = SGD([self.processed_img], lr=5)

            output = self.model(self.processed_img)
            
            target_confidence = functional.softmax(output[0], dim=0)[self.target_label].data.numpy()
            if target_confidence > self.min_confidence:
                print('Generated fooling image with', "{0:.2f}".format(target_confidence),
                        'confidence at', str(i) + 'th iteration.')
                return self.processed_img

            loss = -output[0, self.target_label]
            print('Iteration:', str(i), 'Target confidence', "{0:.4f}".format(target_confidence))
            self.model.zero_grad()
            loss.backward()
            # Clips
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip)
            optimizer.step()
            self.init_img = recreate_image(self.processed_img)

            cv2.imwrite('generated/' + self.org_label + "_" + str(self.target_label) + '.jpg',
                        self.init_img)
        return self.processed_img


def preprocess_image(cv2im, resize=False):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if resize:
        cv2im = cv2.resize(cv2im, (224, 224))
    img_arr = np.float32(cv2im)
    img_arr = np.ascontiguousarray(img_arr[..., ::-1])
    img_arr = img_arr.transpose(2, 0, 1)  # Convert array to D,W,H

    for channel, _ in enumerate(img_arr):
        img_arr[channel] /= 255
        img_arr[channel] -= mean[channel]
        img_arr[channel] /= std[channel]

    img_ten = torch.from_numpy(img_arr).float()

    img_ten.unsqueeze_(0)

    im_as_var = Variable(img_ten, requires_grad=True)
    return im_as_var


def recreate_image(im_as_var):

    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_img = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_img[c] /= reverse_std[c]
        recreated_img[c] -= reverse_mean[c]
    recreated_img[recreated_img > 1] = 1
    recreated_img[recreated_img < 0] = 0
    recreated_img = np.round(recreated_img * 255)

    recreated_img = np.uint8(recreated_img).transpose(1, 2, 0)

    recreated_img = recreated_img[..., ::-1]
    return recreated_img

preprocess = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == '__main__':
    # img_path = "peppers.jpg"
    org_label = "peppers"
    img_path = org_label + ".jpg"
    org_image = cv2.imread(img_path, 1)
    # Define model
    pretrained_model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    
    fooling_target = 812 # space shuttle
    min_confidence = 0.99
    fool = FoolingCNNModel(pretrained_model,
                            org_image,
                            org_label,
                            fooling_target,
                            min_confidence)
    generated_image = fool.generate()

    names = ["peppers", "beagle", "basset", "Siamese"]
    for name in names:
        filename = "generated/" + name + "_812.jpg"
        filename1 = name + ".jpg"
        filename2 = "generated/" + name + "_noise.jpg"
        org = Image.open(filename)
        org1 = Image.open(filename1)
        input_tensor = preprocess(org)
        input_tensor1 = preprocess(org1)
        new_img_PIL = transforms.ToPILImage()(input_tensor1 - input_tensor).convert('RGB')
        new_img_PIL.save(filename2)


