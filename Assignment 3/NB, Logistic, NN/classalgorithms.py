from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import random

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']

    def learn(self, Xtrain, ytrain):
        if self.usecolumnones == False:
            Xtrain = Xtrain[:,0:8]

        self.x0 = []
        self.x1 = []
        for i in range(Xtrain.shape[0]):
            if(ytrain[i]==0):
                self.x0.append(Xtrain[i])
            else:
                self.x1.append(Xtrain[i])

        self.x0 = np.asarray(self.x0).reshape(len(self.x0),Xtrain.shape[1])
        self.x1 = np.asarray(self.x1).reshape(len(self.x1),Xtrain.shape[1])
        self.mean0 = utils.mean(self.x0)
        self.stdev0 = utils.stdev(self.x0)
        self.mean1 = utils.mean(self.x1)
        self.stdev1 = utils.stdev(self.x1)
        self.ymean1 = utils.mean(ytrain)
        self.ymean0 = 1 - self.ymean1

    def predict(self, Xtest):
        if self.usecolumnones == False:
            Xtest = Xtest[:,0:8]

        ytest = np.zeros(Xtest.shape[0])
        for i in range(0,Xtest.shape[0]):
            prob0 = self.ymean0
            prob1 = self.ymean1
            for j in range(0,len(self.mean0)):
                prob0 = prob0 * utils.calculateprob(Xtest[i][j], self.mean0[j], self.stdev0[j])
                prob1 = prob1 * utils.calculateprob(Xtest[i][j], self.mean1[j], self.stdev1[j])

            if(prob0 < prob1):
                ytest[i]=1

        return ytest

class LogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        self.weights = None

        # Set step-size
        self.stepsize = 10

        # Number of repetitions over the dataset
        self.reps = 55

    def learn(self, Xtrain, ytrain):
        ytrain.shape=(len(ytrain),1)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        self.weights1 = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)

        for reps in range(self.reps):
            self.xvec = np.dot(Xtrain, self.weights)
            self.p = utils.sigmoid(self.xvec)
            self.P = np.diagflat(self.p)
            self.weights = self.weights + np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.eye(self.p.size)-self.P),Xtrain)),Xtrain.T),(ytrain-self.p))
            self.weights1[:-1] = self.weights1[:-1] + np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.dot(Xtrain[:,:-1].T,self.P),np.eye(self.p.size)-self.P),Xtrain[:,:-1])),Xtrain[:,:-1].T),(ytrain-self.p))   # self.weights[1:]/Xtrain.shape[0])


    def predict(self, Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

class LogitRegNew(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        self.weights = None

        # Set step-size
        self.stepsize = 10

        # Number of repetitions over the dataset
        self.reps = 5

    def learn(self, Xtrain, ytrain):
        ytrain.shape=(len(ytrain),1)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)

        for reps in range(100):
            self.xvec = np.dot(Xtrain, self.weights)
            self.p = utils.sigmoid(self.xvec)
            self.P = np.diagflat(self.p)
            cos = np.divide(1,(np.sqrt(1+np.square(self.xvec))))
            d = np.multiply(cos,(2*ytrain - self.func(self.xvec)))
            delta = np.dot(Xtrain.T, d)
            self.weights = self.weights + delta;

    def func(self, xvec):
        return np.add(1,np.divide(xvec,(np.sqrt(1+np.square(xvec)))))
        # return np.add(1,np.divide(xvec,(np.sqrt(1+np.square(xvec)))))

    # def dfunc(self, func, ytrain):
        # return np.dot(Xtrain, 2*(1-np.square(func))*(2*ytrain-(1+func)))

    def predict(self, Xtest):
        ytest = self.func(np.dot(Xtest, self.weights))     # utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest

class NeuralNet(Classifier):
    """ Two-layer neural network; need to complete the inherited learn and predict functions """

    def makeMatrix(self, I, J):
        m = []
        for i in range(I):
            m.append([random.uniform(-1,1)]*J)
        return m

    def __init__(self, params=None):
        import utilities as utils
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Set step-size
        self.stepsize = 0.05

        # Number of repetitions over the dataset
        self.reps = 5

        # Create random {0,1} weights to define features
        # self.wi = np.random.randint(2, size=(self.nh, self.ni))*0.0
        # self.wo = np.random.randint(2, size=(self.no, self.nh))*0.0
        self.wi = np.asarray(self.makeMatrix(self.nh, self.ni))
        self.wo = np.asarray(self.makeMatrix(self.no, self.nh+1))


    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """
        for reps in range(self.reps):
            self.loss=0
            self.grad1=0
            self.grad2=0
            for samp in range(2000):
                self.update(Xtrain[samp,:],ytrain[samp])

    def evaluate(self, inputs):
        """ Including this function to show how predictions are made """
        # if inputs.shape[0] != self.ni:
        #     raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')

        # hidden activations
        ah = np.ones((self.nh))
        ah = self.transfer(np.dot(self.wi,inputs))   # (64*9) * (9*1) = (64*1)
        # Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
        ah = np.append([1], ah)

        # output activations
        ao = np.ones((self.no))
        ao = self.transfer(np.dot(self.wo,ah))    # (1*65) * (65*1) = (1*1)
        # ao = NeuralNet.sigmoid(np.dot(self.wo,ah))

        return (ah, ao)

    def update(self, inp, y):
        """ This function needs to be implemented """
        [ah, ycap] = self.evaluate(inp)
        inp = np.reshape(inp.T,(1,9))
        delta = ( -y/ycap + (1-y)/(1-ycap) ) * ycap * (1 - ycap)
        z = np.dot(self.wi, inp.T)  # (64*9) * (9*1)
        delta1 = delta * np.multiply(self.wo[:,1:].T,self.dtransfer(z))    #  (1*64) * (64*1) = (64*1)
        self.wo -= self.stepsize * delta * ah   # 6,
        self.wi -= self.stepsize * np.dot(delta1, inp)   # (64*1) * (1*9)= (64*9)

    # Need to implement predict function, since currently inherits the default
    def predict(self, Xtest):
        ah = utils.sigmoid(np.dot(Xtest, self.wi.T))
        ah = np.hstack((np.ones((ah.shape[0],1)), ah))
        ytest = utils.sigmoid(np.dot(ah,self.wo.T))
        ytest[ytest >= 0.5] = 1
        ytest[ytest < 0.5] = 0
        return ytest
