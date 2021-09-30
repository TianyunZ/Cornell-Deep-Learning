import numpy as np
import matplotlib.pyplot as plt
import random, argparse
import sys
from softmax import Softmax 


class Regression:

    def __init__(self, epochs, learning_rate, batch_size, alpha, model):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.model = model
        self.weight = None

    def train(self, x_train, y_train, x_test, y_test):
        dim = x_train.shape[1]
        self.weight = 0.001 * np.random.rand(dim)
        trainLosses = []
        testLosses = []
        trainMSE = []
        testMSE = []
        for epoch in range(self.epochs):
            trainLoss = self.SGD(x_train, y_train)
            if model == "L1":
                testLoss, _ = self.computeLossL1(x_test, y_test)
            elif model == "L2":
                testLoss, _ = self.computeLossL2(x_test, y_test)
            elif model == "poisson":
                testLoss, _ = self.poissonRegression(x_test, y_test)
            trainLosses.append(trainLoss)
            testLosses.append(testLoss)
            trainMSE.append(musicMSE(x_train.dot(self.weight), y_train))
            testMSE.append(musicMSE(x_test.dot(self.weight), y_test))
            print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainMSE : {:.7f}\t|\tTestMSE: {:.7f}"
                  .format(epoch, trainLoss, testLoss, trainMSE[-1], testMSE[-1]))
        return trainLosses, testLosses, trainMSE, testMSE

    def SGD(self, x, y):
        losses = []
        randInd = random.sample(range(x.shape[0]), x.shape[0])
        x = x[randInd]
        y = y[randInd]
        for i in range(0, x.shape[0], self.batch_size):
            Xbatch = x[i: i + self.batch_size]
            ybatch = y[i: i + self.batch_size]
            loss, dw = self.poissonRegression(Xbatch, ybatch)
            self.weight -= self.learning_rate * dw
            losses.append(loss)
        return np.sum(losses) / len(losses)

    def computeLossL1(self, x, y):
        samples = x.shape[0]
        y_star = np.dot(x, self.weight)
        loss = np.sum((y_star - y) ** 2)
        regLoss = self.alpha * np.linalg.norm(self.weight,ord=1)
        totalLoss = loss + regLoss

        grad = (-2) * np.dot(y-y_star, x) + self.alpha

        return totalLoss/samples, grad

    def computeLossL2(self, x, y):
        samples = x.shape[0]
        scores = np.dot(x, self.weight)
        y_star = scores
        loss = np.sum((y_star - y) ** 2)
        regLoss = self.alpha * np.sum(self.weight ** 2)
        totalLoss = loss + regLoss

        grad = 2 * np.dot(y_star-y, x) + 2*self.alpha*np.linalg.norm(self.weight,ord=1)
        return totalLoss / samples, grad

    def poissonRegression(self, x, y):
        ## loss are less than 0 
        samples = x.shape[0]
        y_star = np.dot(x, self.weight)
        loss = np.exp(y_star) - np.dot(y, y_star)
        totalLoss = np.sum(loss)

        grad = np.sum(np.dot(np.exp(y_star), x) - np.dot(y, x))
        return totalLoss/samples, grad


def loadMusicData(fname, addBias=True):
    data = np.loadtxt(fname, delimiter=',')
    # X1 = data[:463714, 1:]
    # y1 = data[:463714, 0].astype(int)
    # X2 = data[463714:, 1:]
    # y2 = data[463714:, 0].astype(int)
    X1 = data[:800, 1:]
    y1 = data[:800, 0].astype(int)
    X2 = data[800:, 1:]
    y2 = data[800:, 0].astype(int)
    if addBias:
        # yMean = np.mean(y1, axis = 0)
        # y1 = y1 - int(yMean)
        # y2 = y2 - int(yMean)
        y1 = y1 - 1922
        y2 = y2 - 1922
    return y1, X1, y2, X2

def musicMSE(pred, gt):
    mse = 0
    for i in range(len(pred)):
        mse += (pred[i]-gt[i])*(pred[i]-gt[i])
    return mse/len(pred)

def analyseData(trainY, testY):
    data = np.append(trainY, testY)
    plt.hist(data, bins = 90)
    plt.title("Histogram of the labels in the train and test set")
    plt.xlabel("Label of Year")
    plt.ylabel("Number")
    plt.show()

def createFig(trainLosses, testLosses, trainMSE, testMSE):
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="Train loss")
    plt.plot(testLosses, label="Test loss")
    plt.legend(loc='best')
    plt.title("Average Train Cross Entropy Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Entropy Loss")

    plt.subplot(1, 2, 2)
    plt.plot(trainMSE, label="Train MSE")
    plt.plot(testMSE, label="Test MSE")
    plt.legend(loc='best')
    plt.title("Mean MSE varying with Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Mean MSE")

    plt.show()


# fname = "YearPredictionMSD.txt"
# # fname = "demoMSD.txt"
# trainYears, trainFeat, testYears, testFeat = loadMusicData(fname, True)
# print(type(trainYears))
# print("---", len(trainYears), "----")

# analyseData(trainYears, testYears)
# analyseData(trainFeat[:, 0], testFeat[:, 0])

## number of features
# feat_num = {}
# for line in trainFeat:
#     if len(line) in feat_num.keys():
#         feat_num[len(line)] += 1
#     else:
#         feat_num[len(line)] = 1
# print(feat_num)

# mx = {}
# mn = {}
# mx_list = []
# mn_list = []
# for i in range(90):
#     data = np.append(trainFeat[:, i], testFeat[:, i])
#     mx[i] = max(data)
#     mn[i] = min(data)
#     mx_list.append(max(data))
#     mn_list.append(min(data))
# # print("max:", mx)
# # print("min:", mn)
# print("max:", max(mx_list))
# print("min:", min(mn_list))
# data = np.append(trainYears, testYears)
# print("years:", max(data), min(data))
# year_dict = {}
# for v in data:
#     if v in year_dict.keys():
#         year_dict[v] += 1
#     else:
#         year_dict[v] = 1
# print(year_dict)
# print(len(year_dict))

# predYears = np.ones(len(testYears))
# print(predYears[:10]*2007)
# # mse = musicMSE(predYears*1998, testYears)
# mse = musicMSE(predYears*2007, testYears)
# print(mse)


if __name__ == '__main__':
    """
    Main method
    """
    # fname = "YearPredictionMSD.txt"
    fname = "demoMSD.txt"
    trainYears, trainFeat, testYears, testFeat = loadMusicData(fname, False)
    
    # print(int(yearMean))
    # sys.exit()
    
    yearMean = np.mean(trainYears, axis = 0)
    yearStd = np.std(trainYears, axis=0)
    trainYears = np.subtract(trainYears, 1992) / yearStd
    testYears = np.subtract(testYears, 1992) / yearStd
    print("trainYears", trainYears)
    print("testYears", testYears)

    allFeat = np.append(trainFeat,testFeat,axis=0)
    featMean = np.mean(trainFeat, axis = 0)
    featStd = np.std(trainFeat, axis=0)
    trainFeat = np.subtract(trainFeat, featMean) / featStd
    testFeat = np.subtract(testFeat, featMean) / featStd


    print(type(trainYears))
    print("---", len(trainYears), "----")

    epochs = 50
    learning_rate = 0.000001
    batch_size = 10
    alpha = 0.01
    model = "L2"

    ## Problem 3-(2~4)
    rg = Regression(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, 
                    alpha=alpha, model=model)
    trainLosses, testLosses, trainMSE, testMSE = rg.train(trainFeat, trainYears, testFeat, testYears)
    
    createFig(trainLosses, testLosses, trainMSE, testMSE)
    
    ## Problem 3-(5)
    # sm = Softmax(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, 
    #               alpha=alpha, momentum=momentum)
    # trainLosses, testLosses, trainMSE, testMSE = sm.train(trainFeat, trainYears, testFeat, testYears)
    # createFig(trainLosses, testLosses, trainMSE, testMSE)