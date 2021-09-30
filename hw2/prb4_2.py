import torch
import sys
import numpy as np
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

from PIL import Image
from PIL import ImageOps
from torchvision import transforms

# filename = "Siamese.jpg"
name = ["peppers", "beagle", "basset", "Siamese"]
for obj in name:
    print("###" + obj + ">>>")
    # obj = "Siamese"
    filename = 'generated/' + obj + "_" + str(812) + '.jpg'
    # filename = obj + ".jpg"

    input_image = Image.open(filename)
    # input_image0 = Image.open(filename0)

    org = Image.open(filename)
    # Crop
    # w,h = org.size
    # dw = round(0.1*w)
    # dh = round(0.1*h)
    # org = org.crop((dw, dh, w-dw, h-dh))
    # Mirror
    # org = ImageOps.mirror(org)
    # Rotate
    # org = org.rotate(30)
    # Grayscale
    # org = np.mean(org, axis=2)
    org = org.convert('L')
    org = org.convert('RGB')
    # org.show()
    # sys.exit()
    org.save("p4_img/" + "gray_" + obj + ".jpg")
    # sys.exit()
    # org.show()
    # noise = Image.open('generated/' + obj + "_" + "noise" + '.jpg')
    # input_image = org + noise
    
    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop((round(0.8*w), round(0.8*h))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(org)
    # input_tensor = preprocess(input_image)
    # input_tensor0 = preprocess(input_image0)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top3_prob, top3_catid = torch.topk(probabilities, 3)
    for i in range(top3_prob.size(0)):
        print(categories[top3_catid[i]], top3_prob[i].item())
