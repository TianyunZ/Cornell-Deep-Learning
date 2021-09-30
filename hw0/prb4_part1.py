import pickle
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
    return dict

file = "data_batch_1"
dict = unpickle(file)

img = [[[0 for i in range(3)] for j in range(32)] for k in range(32)]
C = [0]*10

for k in range(dict["data"].shape[0]):
    label = dict["labels"][k]
    if C[label] == 3:
        continue
    C[label] += 1
    srcImg = dict["data"][k]
    
    for i in range(32):
        for j in range(32):
            for c in range(3):
                t = 1024*c + 32*i + j
                img[i][j][c] = srcImg[t]
    plt.imshow(img)
    plt.title(f"Training example #{label}-{C[label]}")
    plt.axis('off')
    plt.show()
