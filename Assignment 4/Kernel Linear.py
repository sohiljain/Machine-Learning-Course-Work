
# coding: utf-8

# In[55]:

import csv
import random
import math
import numpy as np

import classalgorithms as algs

def splitdataset(dataset, trainsize=4500, testsize=1500, testfile=None):
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

trainset, testset = loadsusy()
for reps in [x*5 for x in range(1,10)]:
    for regwgt in [(10**x) for x in range(-3,1)]:
        weights = None
        # regwgt = 0.01

        xtrain = trainset[0]
        ytrain = trainset[1]

        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        c = []
        for i in np.random.randint(0,xtrain.shape[0],60):
            c.append(xtrain[i])

        c = np.array(c)

        K = []
        for i in range(xtrain.shape[0]):
            temp = []
            for j in range(c.shape[0]):
                temp.append(np.dot(xtrain[i],c[j].T))
            K.append(temp)
        K = np.array(K)

        numsamples = xtrain.shape[0]
        # weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(xtrain.T,xtrain)/numsamples,regwgt*np.identity(xtrain.shape[1]))), xtrain.T),yt)/numsamples
        weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(K.T,K)/numsamples,regwgt*np.identity(K.shape[1]))), K.T),yt)/numsamples

        Ktest = []
        for i in range(testset[0].shape[0]):
            temp = []
            for j in range(c.shape[0]):
                temp.append(np.dot(testset[0][i],c[j].T))
            Ktest.append(temp)

        ytest = np.dot(Ktest, weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0

        accuracy = getaccuracy(testset[1], ytest)
        print "Accuracy for kernel size: " + str(reps) + ", regularizer weight: " + str(regwgt) + " = " + str(accuracy)