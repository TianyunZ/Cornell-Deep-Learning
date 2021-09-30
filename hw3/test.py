
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from PIL import Image

to_image = transforms.ToPILImage()
# images = [np.array(to_image(i)) for i in images]
# np.save("images_list", images)
images = np.load("models/images_list_5_1.npy")
# print(images)
# c = images[49][:100,:100,:]
# print(c.shape)
# print(len(images))
k_list = [0,25,49]
for k in k_list:
    im = Image.fromarray(images[k])
    im.save("models/prb5_img/epoch_1_"+str(k)+".jpg")

g_losses = np.load("2_1_g_losses.npy")
d_losses = np.load("2_1_d_losses.npy")
plt.plot(g_losses, label='Generator_Losses')
plt.plot(d_losses, label='Discriminator Losses')
plt.legend()
plt.savefig('loss.png')


model = torch.load("models/p5_discriminator.ckpt")

