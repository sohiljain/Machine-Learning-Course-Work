import csv
import random
import math
import numpy as np
from scipy.stats import ttest_ind

import classalgorithms as algs

def splitdataset(dataset, trainsize=500, testsize=300, testfile=None):
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
    correct = 0.0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')
    np.random.shuffle(dataset)
    trainset, testset = splitdataset(dataset)
    return trainset,testset

def k_fold_cross_validation(X, K, randomise = False):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise: from random import shuffle; X=list(X); shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

if __name__ == '__main__':
    # trainset, testset = loadsusy()
    # trainset, testset = loadmadelon()
    accset = {}
    accset['Linear Regression'] = []
    accset['Logistic Regression'] = []
    mp = {}
    mp['Linear Regression'] = []
    mp['Logistic Regression'] = []
    dataset = np.genfromtxt('susysubset.csv', delimiter=',')
    np.random.shuffle(dataset)

    mp_perf = []
    for i in range(7):
        extdataset = dataset[i*1200:(i+1)*1200]
        classalgs = {
            'Linear Regression': algs.LinearRegressionClass(),
            'Logistic Regression': algs.LogitReg()
        }

        maxaccuracy = 0
        bestmp = 0
        for learnername, learner in classalgs.iteritems():
            for trainset, testset in k_fold_cross_validation(extdataset, K=10):
                trainset = np.array(trainset)
                testset = np.array(testset)
                numinputs = -1
                Xtrain = trainset[:,0:numinputs]
                ytrain = trainset[:,numinputs]
                Xtest = testset[:,0:numinputs]
                ytest = testset[:,numinputs]

                # Add a column of ones; done after to avoid modifying entire dataset
                Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
                Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))

                trainset = (Xtrain,ytrain)
                testset = (Xtest,ytest)

                for regwgt in [10**x for x in range(-5,1)]:

                    learner.par({'regwgt': regwgt})
                    learner.learn(trainset[0], trainset[1])
                    predictions = learner.predict(testset[0])
                    accuracy = getaccuracy(testset[1], predictions)
                    print(learnername + ': ' + str(accuracy))

                    if maxaccuracy < accuracy:
                        maxaccuracy = accuracy
                        bestmp = regwgt

            accset[learnername].append(maxaccuracy)
            mp[learnername].append(bestmp)

print("Accuracy set : " + str(accset))
print("Metaparameter set : " + str(mp))
print("Mean Accuracy Linear : "+ str(np.average(accset['Linear Regression'])))
print("Mean Accuracy Logistic : "+ str(np.average(accset['Logistic Regression'])))
print(np.var(accset['Linear Regression']))
print(np.var(accset['Logistic Regression']))

statistic, pvalue = ttest_ind(accset['Linear Regression'], accset['Logistic Regression'], equal_var=False)
print "statistic = %.5f and pvalue = %.10f" % (statistic, pvalue)
statistic, pvalue = ttest_ind(accset['Linear Regression'], accset['Logistic Regression'], equal_var=True)
print "statistic = %.5f and pvalue = %.10f" % (statistic, pvalue)
