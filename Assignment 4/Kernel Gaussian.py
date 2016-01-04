
# coding: utf-8

# In[55]:

import csv
import random
import math
import numpy as np

import classalgorithms as algs

def splitdataset(dataset, trainsize=2500, testsize=500, testfile=None):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))

    return ((Xtrain,ytrain), (Xtest,ytest))


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')
    trainset, testset = splitdataset(dataset)
    return trainset,testset
    return trainset,testset

def linearKernel(X,C):
    K = []
    for i in range(X.shape[0]):
        temp = []
        for j in range(C.shape[0]):
            temp.append(np.dot(X[i],C[j].T))
        K.append(temp)
    return np.array(K)

def polynomialKernel(X,C):
    K = []
    const = 3
    for i in range(X.shape[0]):
        temp = []
        for j in range(C.shape[0]):
            temp.append((np.dot(X[i],C[j].T)+const)**3)
        K.append(temp)
    return np.array(K)

def gaussianKernel(X,C,sigma):
    K = []
    for i in range(X.shape[0]):
        temp = []
        for j in range(C.shape[0]):
            x = np.exp(-(np.linalg.norm(X[i]-C[j])**2)/(2*(sigma**2)))
            temp.append(x)
        K.append(temp)
    return np.array(K)

trainset, testset = loadsusy()
xtrain = trainset[0]
ytrain = trainset[1]

yt = np.copy(ytrain)
yt[yt == 0] = -1

c = []
for i in np.random.randint(0,xtrain.shape[0],60):
    c.append(xtrain[i])
c = np.array(c)

for reps in [x*6 for x in range(10,7,-2)]:
    for regwgt in [(10**x) for x in range(-3,-1)]:
        for sigma in [x for x in range(1,20,5)]:
            weights = None

            K = gaussianKernel(trainset[0], c, sigma)
            numsamples = xtrain.shape[0]
            weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(K.T,K)/numsamples,regwgt*np.identity(K.shape[1]))), K.T),yt)/numsamples
            Ktest = gaussianKernel(testset[0], c, sigma)
            ytest = np.dot(Ktest, weights)
            ytest[ytest > 0] = 1
            ytest[ytest < 0] = 0
            accuracy = getaccuracy(testset[1], ytest)
            print "Accuracy for Gaus kernel size: " + str(reps) + ", regularizer weight: " + str(regwgt) + ", sigma: " + str(sigma) + " = " + str(accuracy)