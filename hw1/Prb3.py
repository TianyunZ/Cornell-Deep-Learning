import pickle
import matplotlib.pyplot as plt
import sys
import numpy as np
import itertools
from softmax import Softmax
from sklearn.metrics import confusion_matrix

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
    return dict

def loadData(file):
    dict = unpickle(file)
    X = []
    y = []

    for k in range(dict["data"].shape[0]):
        label = dict["labels"][k]
        srcImg = dict["data"][k]
        X.append(srcImg)
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def createFig(trainLosses):
    plt.plot(trainLosses, label="Train loss")
    plt.legend(loc='best')
    plt.title("Train Loss varying with Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def getConfusionMatrix(actualLabel, predictedLabel, numOfClass):
    confMtrx =[]
    for _ in range(numOfClass):
        confMtrx.append([])
        for _ in range(numOfClass):
            confMtrx[-1].append(0)

    for sampleNum in range(actualLabel.shape[0]):
        confMtrx[int(actualLabel[sampleNum])][int(predictedLabel[sampleNum])] += 1
    confMtrx = np.array(confMtrx)
    return confMtrx

def plotConfusionMatrix(s81, xTest, actualLabel, classes, normalize=False,
                        title='Confusion matrix', cmap=plt.cm.Blues):
    predY = s81.predict(xTest)
    predY = predY.reshape((-1, 1))
    confMtrx = getConfusionMatrix(actualLabel, predY, 10)
    if normalize:
        confMtrx = confMtrx.astype('float') / confMtrx.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    print(confMtrx)

    plt.imshow(confMtrx, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confMtrx.max() / 2.
    for i, j in itertools.product(range(confMtrx.shape[0]), range(confMtrx.shape[1])):
        plt.text(j, i, format(confMtrx[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confMtrx[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    trainX = []
    trainY = []
    # for i in range(1, 6):
    #     x, y = loadData("cifar-10-batches-py/data_batch_"+str(i))
    #     trainX = trainX + x
    #     trainY = trainY + y
    trainX, trainY = loadData("cifar-10-batches-py/data_batch_1")
    testX, testY = loadData("cifar-10-batches-py/test_batch")
    print(trainX.shape)
    trainX = np.dstack((trainX[:, :1024], trainX[:, 1024:2048], trainX[:, 2048:])) / 255.
    trainX = np.reshape(trainX, [-1, 32, 32, 3])
    testX = np.dstack((testX[:, :1024], testX[:, 1024:2048], testX[:, 2048:])) / 255.
    testX = np.reshape(testX, [-1, 32, 32, 3])
    print(trainX.shape)
    # print(trainX)
    # sys.exit()

    featMean = np.mean(trainX, axis = 0)
    featStd = np.std(trainX, axis=0)
    testMean = np.mean(testX, axis = 0)
    testStd = np.std(testX, axis=0)
    trainX = np.subtract(trainX, featMean) / featStd
    testX = np.subtract(testX, testMean) / testStd

    epochs = 100
    learningRate = 0.001
    batchSize = 20
    wtHyper = 0.0001
    momentum = 0.005

    sm = Softmax(epochs=epochs, learningRate=learningRate, batchSize=batchSize,
                 wtHyper=wtHyper, momentum=momentum)
    trainLosses, testLosses, trainAcc, testAcc = sm.train(trainX, trainY, testX, testY)
    createFig(trainLosses)
    # confusion_matrix(y_true, y_pred)
    plotConfusionMatrix(sm, testX, testY, "0123456789",
                        normalize=True, title='Normalized confusion matrix')
    plt.show()
