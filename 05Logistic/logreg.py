import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math


filepath1 = './lin_sample.csv'
filepath2 = './unlin_sample.csv'
weights1 = './weights1'
weights2 = './weights2'
epoches = 3000
alpha = 0.001


def readfile(filepath):
    df = pd.read_csv(filepath)
    alt = np.array(df[['x', 'y', 'class']])
    dataMat = []
    labelMat = []
    for data in alt:
        if data[2] == 0:
            dataMat.append([1.0,float(data[0]),float(data[1])])
            labelMat.append(0)
        else:
            dataMat.append([1.0,float(data[0]),float(data[1])])
            labelMat.append(1)
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+math.exp(-inX))


def gradAscent(dataMatrix, labelMat,alpha,weights):
    total_error = 0
    m = np.shape(dataMatrix)[0]
    for i in range(m):
        h = sigmoid(dataMatrix[i]*weights)
        error = (labelMat[i] - h)
        total_error += error[0,0] ** 2
        weights = weights + alpha * dataMatrix[i].transpose() * error
    return weights,total_error

def plotBestFit(filepath,weights):

    datas, labels = readfile(filepath)
    datas = np.array(datas)
    n = np.shape(datas)[0]
    x1,y1,x2,y2 = [],[],[],[]
    class1right , class0right = 0,0
    for i in range(n):
        if labels[i] == 1:
            x1.append(datas[i, 1])
            y1.append(datas[i, 2])

            if classifyVector(np.array([1.0,datas[i, 1],datas[i, 2]]),np.array(weights).transpose()) == 1:
                class1right += 1
        else:
            x2.append(datas[i, 1])
            y2.append(datas[i, 2])
            if classifyVector(np.array([1.0,datas[i, 1],datas[i, 2]]),np.array(weights).transpose()) == 0:
                class0right += 1
    print(class1right,class0right)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='blue')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    ax.text(-3,12,'Blue: {}/10\nRed: {}/10'.format(class0right,class1right))
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classifyVector(inX, weights):
    '''
    分类函数
    '''
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def drawError(errors,iflinear):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0,len(errors),1)
    ax.plot(x,np.array(errors))
    plt.xlabel('epoch')
    plt.ylabel('error')
    if iflinear:
        plt.title('linear')
    else:
        plt.title('unlinear')
    plt.show()

def getWeights(weightfile):
    return np.genfromtxt(weightfile, delimiter=',')

def train(iflinear=True):
    if iflinear:
        filepath = filepath1
        weightfile = weights1
    else:
        filepath = filepath2
        weightfile = weights2

    datas, labels = readfile(filepath)
    datas = np.mat(datas)
    labels = np.mat(labels).transpose()
    m, n = np.shape(datas)
    errors = []
    weights = np.ones((n,1))
    for _ in range(epoches):
        weights,error = gradAscent(datas,labels,alpha,weights)
        # print(weights)
        errors.append(error)

    weights = weights.getA()
    print(weights)
    np.savetxt(weightfile, np.reshape(weights, (-1, 3)), delimiter=",")
    drawError(errors,iflinear)

def test(iflinear=True):
    if iflinear:
        filepath = filepath1
        weightfile = weights1
    else:
        filepath = filepath2
        weightfile = weights2
    weights = getWeights(weightfile)
    print(weights)

    # draw the plot

    plotBestFit(filepath,weights)

def main():
    #linear
    train()
    test()
    #
    #unlinear
    train(False)
    test(False)


if __name__=='__main__':
    main()
