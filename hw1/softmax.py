import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys


class Softmax:

    def __init__(self, epochs, learning_rate, batch_size, alpha, momentum):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.momentum = momentum
        self.vel = None
        self.weight = None

    def train(self, x_train, y_train, x_test, y_test):
        dim = x_train.shape[1]
        label = np.unique(y_train)
        classesNum = len(label)
        # classesNum = 90
        yTrainOneHot = self.toOneHot(y_train, classesNum)
        yTestOneHot = self.toOneHot(y_test, classesNum)
        self.weight = 0.001 * np.random.rand(dim, classesNum)
        self.vel = np.zeros(self.weight.shape)
        trainLosses = []
        testLosses = []
        trainAccurcy = []
        testAccurcy = []
        # trainMSE = []
        # testMSE = []
        for epoch in range(self.epochs):
            trainLoss = self.SGDMomentum(x_train, yTrainOneHot)
            testLoss, _ = self.computeLoss(x_test, yTestOneHot)
            trainLosses.append(trainLoss)
            testLosses.append(testLoss)
            trainAccurcy.append(self.computeAccuracy(x_train, y_train))
            testAccurcy.append(self.computeAccuracy(x_test, y_test))
            # trainMSE.append(self.musicMSE(x_train, y_train))
            # testMSE.append(self.musicMSE(x_test, y_test))
            print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainAcc : {:.7f}\t|\tTestAcc: {:.7f}"
                  .format(epoch, trainLoss, testLoss, trainAccurcy[-1], testAccurcy[-1]))
            # print("{:d}\t->\tTrainL : {:.7f}\t|\tTestL : {:.7f}\t|\tTrainMSE : {:.7f}\t|\tTestMSE: {:.7f}"
                #   .format(e, trainLoss, testLoss, trainMSE[-1], testMSE[-1]))
        # return trainLosses, testLosses, trainMSE, testMSE
        return trainLosses, testLosses, trainAccurcy, testAccurcy

    def toOneHot(self, y, classesNum):
        y = np.asarray(y, dtype='int32')
        if len(y) > 1:
            y = y.reshape(-1)
        yMx = np.zeros((len(y), classesNum))
        yMx[np.arange(len(y)), y] = 1
        return yMx

    def computeSoftmaxProb(self, scores):
        scores -= np.max(scores)
        prob = (np.exp(scores).T / np.sum(np.exp(scores), axis=1)).T
        return prob

    def computeLoss(self, x, y):
        # add L2 regularization
        samples = x.shape[0]
        temp = np.dot(x, self.weight)
        prob = self.computeSoftmaxProb(temp)
        loss = -np.log(np.max(prob)) * y
        L2Loss = (1/2) * self.alpha * np.sum(self.weight * self.weight)
        totalLoss = (np.sum(loss) / samples) + L2Loss
        grad = ((-1 / samples) * np.dot(x.T, (y - prob))) + (self.alpha * self.weight)
        return totalLoss, grad

    def computeAccuracy(self, x, y):
        pred = self.predict(x)
        pred = pred.reshape((-1, 1))
        return np.mean(np.equal(y, pred))
    
    def SGDMomentum(self, x, y):
        losses = []
        randInd = random.sample(range(x.shape[0]), x.shape[0])
        x = x[randInd]
        y = y[randInd]
        for i in range(0, x.shape[0], self.batch_size):
            Xbatch = x[i:i+self.batch_size]
            ybatch = y[i:i+self.batch_size]
            loss, dw = self.computeLoss(Xbatch, ybatch)
            self.vel = (self.momentum * self.vel) + (self.learning_rate * dw)
            self.weight -= self.vel
            losses.append(loss)
        return np.sum(losses) / len(losses)

    def predict(self, x):
        return np.argmax(x.dot(self.weight), 1)

    def musicMSE(self, x, gt):
        mse = 0
        samples = x.shape[0]
        pred = self.predict(x)
        for i in range(len(pred)):
            mse += (pred[i]-gt[i])*(pred[i]-gt[i])
        return mse/samples


def createFig(trainLosses, testLosses, trainAccurcy, testAccurcy):
    plt.subplot(1, 2, 1)
    plt.plot(trainLosses, label="Train loss")
    plt.plot(testLosses, label="Test loss")
    plt.legend(loc='best')
    plt.title("Loss varying with Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(trainAccurcy, label="Train Accuracy")
    plt.plot(testAccurcy, label="Test Accuracy")
    plt.legend(loc='best')
    plt.title("Mean per class Accuracy varying with Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean per class Accuracy")
    plt.show()

def loadData(filename):
    data = np.loadtxt(filename)
    np.random.shuffle(data)
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    y = np.reshape(y, (-1, 1))
    y -= 1
    return X, y

def normalizeData(x):
    return np.subtract(np.dot(x, 2), 1)


def displayDecBoundary(x, y):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x_ax, y_ax = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    arr = np.array([x_ax.ravel(), y_ax.ravel()])
    scores = np.dot(arr.T, sm.weight)
    prob = sm.computeSoftmaxProb(scores)
    Z = np.argmax(prob, axis=1) + 1

    Z = Z.reshape(x_ax.shape)
    plt.contourf(x_ax, y_ax, Z, alpha=0.8)

    markers = ('*', '+', 'x')
    colors = ('crimson', 'green', 'orange')
    colorMap = ListedColormap(colors)

    for idx, cl in enumerate(np.unique(y)):
        xBasedOnLabel = x[np.where(y[:,0] == cl)]
        plt.scatter(x=xBasedOnLabel[:, 0], y=xBasedOnLabel[:, 1], c=colorMap(idx),
                    marker=markers[idx], label=cl)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary of Iris Dataset with Softmax Classifier")
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':

    trainX, trainY = loadData("iris-train.txt")
    testX, testY = loadData("iris-test.txt")

    trainX = normalizeData(trainX)
    testX = normalizeData(testX)

    epochs = 1000
    learning_rate = 0.01
    batch_size = 8
    alpha = 0.001
    momentum = 0.1


    sm = Softmax(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size,
                alpha=alpha, momentum=momentum)
    trainLosses, testLosses, trainAccurcy, testAccurcy = sm.train(trainX, trainY, testX, testY)
    createFig(trainLosses, testLosses, trainAccurcy, testAccurcy)
    displayDecBoundary(trainX, trainY)
    displayDecBoundary(testX, testY)