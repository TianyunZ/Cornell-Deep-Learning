import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# dim=4
# train_data_size = 5633
# df = pd.read_csv("test.csv")
# data = df.values
# norm = np.sqrt(np.sum(data*data, axis=0))
# data_list = data/norm
# print(data_list)
# df = pd.DataFrame(data_list)
# df.to_csv('normed_test.csv')
# print(train_data_list)
# power_list = [0]*dim
# # temp = []
# temp = [[0 for i in range(dim)] for j in range(train_data_size)]
# for index, row in train_data_list.iteritems():
#     print(row)
#     power = 0
#     for x in row:
#         power += x*x
#     L2_norm = pow(power, 0.5)
#     for x in row:
#         temp.append(x/L2_norm)
    # sys.exit()
    # for i in range(row.size()):
    #     power_list[i] += row[i]*row[i]

# Prob 3
array1 = []
with open("prob3_1.txt", "r") as f:
    for line in f.readlines():
        a = line.split(',')[1]
        b = a.split(': ')
        c = b[1].split('\n')
        array1.append(float(c[0]))

array2 = []
with open("prob3_2.txt", "r") as f:
    for line in f.readlines():
        a = line.split(',')[1]
        b = a.split(': ')
        c = b[1].split('\n')
        array2.append(float(c[0]))

# plt.subplot(1, 2, 1)
plt.plot(array1, label="Loss without Batch Norm")
plt.plot(array2, label="Loss with Batch Norm")
plt.legend(loc='best')
plt.title("Loss varying with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Prob 2
array = []
with open("prb2_1.txt", "r") as f:
    for line in f.readlines():
        # print(line)
        # sys.exit()
        array.append(float(line))

plt.plot(array, label="Mean-per-class Accuracy")
plt.title("Mean-per-class Accuracy varying with Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()