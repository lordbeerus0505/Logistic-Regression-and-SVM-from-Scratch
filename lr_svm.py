""" 
Please put your code for this question in a le called lr svm.py. This script should take three
command-line-arguments as input as described below:
1. trainingDataFilename: the set of data that will be used to train your algorithms (e.g., train-
ingSet.csv).
2. testDataFilename: the set of data that will be used to test your algorithms (e.g., testSet.csv).
3. modelIdx: an integer to specify the model to use for classication (LR= 1 and SVM= 2).
"""


import sys
import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


class SupportVectorMachine(object):
    """ 
    Write a function named svm(trainingSet, testSet) which takes the training dataset and
    the testing dataset as input parameters. The purpose of this function is to train a linear SVM
    classier using the data in the training dataset, and then test the classier's performance on
    the testing dataset.
    Use the following setup for training the SVM: (1) Use hinge loss. Optimize with subgradient
    descent, using an initial weight of all zeros, a step size of 0:5 and a regularization parameter
    of lambdaValue = 0:01. (2) Stop optimization after a maximum number of iterations max = 500, or
    when the L2 norm of the dierence between new and old weights is smaller than the threshold
    tol = 1e - 6, whichever is reached rst. Print the classier's accuracy on both the training
    dataset and the testing dataset (rounded to two decimals).
    """
    def __init__(self, learning_rate = 0.5, numIterations = 500, penalty = None, lambdaValue = 0.01, tol = 10**-6):
        self.learningRate = learning_rate
        self.numIterations = numIterations
        self.penalty = penalty
        self.lambdaValue = lambdaValue
        self.tol = tol
    
    def svmGradientCost(self, W, X_batch, Y_batch):
        yi_hat = np.dot(X_batch, W)
        product = yi_hat*Y_batch
        
        netResult = np.where(product<1,Y_batch,0) 
        # when mistake the product is less than 1.
        # since 0 into anything anyway 0 can put it here and then multiply later
        netResult = np.tile(netResult.transpose(),(X_batch.shape[1],1))
        delta_ji = netResult.T*X_batch
        
        delta = self.lambdaValue*W - delta_ji
        delta = np.sum(delta, axis=0)/len(Y_batch)
        return delta


    def svmTrain(self, X, Y):
        #Change all X,Y to xtrain y train
        weights = np.zeros(X.shape[1])
        svmObj = SupportVectorMachine()

        # Gradient descent
        for _ in range(self.numIterations):
            # print("Epoch %s completed"%epoch)
            descent = svmObj.svmGradientCost(weights, X, Y)
            delta_w = self.learningRate * descent

            if norm(delta_w) >= self.tol: 
                weights -= delta_w
            else:
                break
        return weights

class LogisticRegression(object):    
    """ 
    Write a function named lr(trainingSet, testSet) which takes the training dataset and the
    testing dataset as input parameters. The purpose of this function is to train a logistic regres-
    sion classier using the data in the training dataset, and then test the classier's performance
    on the testing dataset.
    Use the following setup for training the logistic regression classier: (1) Use L2 regularization,
    with lambdaValue = 0:01. Optimize with gradient descent, using an initial weight vector of all zeros and a
    step size of 0:01. (2) Stop optimization after a maximum number of iterations max = 500, or when the L2 norm of the dierence between new and old weights is smaller than the threshold
    tol = 1e - 6, whichever is reached rst. Print the classier's accuracy on both the training
    dataset and the testing dataset (rounded to two decimals).
    """
    def __init__(self, learningRate, numIterations = 10, penalty = None, lambdaValue = 0.01):
        
        self.learningRate = learningRate
        self.numIterations = numIterations
        self.penalty = penalty
        self.lambdaValue = lambdaValue
        
    def train(self, X_train, y_train, tol = 10 ** -4):
        import copy
        
        # +1 for the bias term, w0 in weight vector
        
        LRobj = LogisticRegression(learningRate = 0.001, numIterations = 5, penalty = 'L2', lambdaValue= 0.01)



        self.weights = np.zeros(np.shape(X_train)[1] + 1) 
        X_train_copy = X_train[:]
        X_train = np.c_[np.ones([np.shape(X_train)[0], 1]), X_train]
        self.costs = []
        for i in range(self.numIterations):
            z = np.dot(X_train, self.weights)
            errors = - y_train + logistic_func(z) # using notation in slides - yi + yi^

            if self.penalty is not None:    
                # Not producing single number but a list of numbers 
                weightAdj = np.true_divide(self.weights, self.lambdaValue, dtype=np.float64)
                prod = np.dot(errors, X_train) + weightAdj
                delta_w = self.learningRate * prod
            else:
                delta_w = self.learningRate * np.dot(errors, X_train)
            
            self.iterationsPerformed = i

            # Find L2 norm and update only if greater than tolerance

            if norm(delta_w) >= tol: 
                #weight update
                self.weights -= delta_w   
                LRobj.weights = self.weights 
                pred, prob = LRobj.predict(X_train_copy, 0.5)                       

            else:
                break
            
        return self
                    
    def predict(self, X_test, pi = 0.5):
        z = self.weights[0] + np.dot(X_test, self.weights[1:])     
        probs = np.array([logistic_func(i) for i in z])
        predictions = np.where(probs >= pi, 1, 0)
        return predictions, probs
        
    def performanceEval(self, predictions, y_test):
        count = 0
        diff = np.subtract(predictions, y_test)
        return np.sum(np.abs(diff))/ len(predictions)
        

def logistic_func(z):   
    return 1 / (1 + np.exp(-z))  
    
def logLiklihood(z, y):
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z))))

def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def reg_logLiklihood(x, weights, y, lambdaValue):
    z = np.dot(x, weights) 
    reg_term = (1 / (2 * lambdaValue)) * np.dot(weights.T, weights)
    return -1 * np.sum((y * np.log(logistic_func(z))) + ((1 - y) * np.log(1 - logistic_func(z)))) + reg_term

def svm(trainingSet, testSet):
    Y = trainingSet.loc[:, 'decision']
    X = trainingSet.drop(['decision'], axis=1)
    Y= np.where(Y==0,-1,Y)
    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)
    
    X_train = X[:]
    y_train = Y[:]
    y_test = testSet.loc[:, 'decision']
    y_test= np.where(y_test==0,-1,y_test)
    X_test = testSet.drop(['decision'], axis=1)

    X_test.insert(loc=len(X_test.columns), column='intercept', value=1)
    svm = SupportVectorMachine(learning_rate = 0.5, numIterations = 500, penalty = 'L2', lambdaValue = 0.01)
    W = svm.svmTrain(X_train.to_numpy(), y_train)


    # testing the model
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)
    print('Training Accuracy SVM: %.5f'%svmPerformance(y_train, y_train_predicted))


    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)
    print('Test Accuracy SVM: %.5f'%svmPerformance(y_test, y_test_predicted))


def lr(trainingSet, testSet):

    y_train = trainingSet['decision']
    X_train = trainingSet.drop(['decision'], axis=1)
    
    """ # Trying out with simple data to see that the model actually works
    X_train = [[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1],
     [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0],
     [1,1,0,1], [1,1,1,0], [1,1,1,1]]
    # if b,d true
    y_train = [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1]
    y_train = pd.Series( (v for v in y_train) ) 
    # Output came to 1.00 accuracy on train set without normalization
    """
   
    #######################################################################
    # CONSIDER SHUFFLING DATASET EVERY EPOCH ##############################
    # RESET ALL C WITH LAMBDA AND FLIP THE SIGNS ##########################
    #######################################################################
    LR = LogisticRegression(learningRate = 0.01, numIterations = 500, penalty = 'L2', lambdaValue= 100)  
    LR.train(X_train, y_train, tol = 10 ** -6)

    y_test = testSet['decision']
    X_test = testSet.drop(['decision'], axis=1)

    predictions, probs = LR.predict(X_train, 0.5)
    performance = lrPerformance(predictions, y_train)
    print('Training Accuracy LR: %.5f'%performance)

    predictions, probs = LR.predict(X_test, 0.5)
    performance = lrPerformance(predictions, y_test)
    print('Test Accuracy LR: %.5f'%performance)

def svm_crossValidate(trainingSet, testSet):
    Y = trainingSet.loc[:, 'decision']
    X = trainingSet.drop(['decision'], axis=1)
    Y= np.where(Y==0,-1,Y)
    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)
    
    X_train = X[:]
    y_train = Y[:]
    y_test = testSet.loc[:, 'decision']
    y_test= np.where(y_test==0,-1,y_test)
    X_test = testSet.drop(['decision'], axis=1)

    X_test.insert(loc=len(X_test.columns), column='intercept', value=1)
    svm = SupportVectorMachine(learning_rate = 0.5, numIterations = 500, penalty = 'L2', lambdaValue = 0.01)
    W = svm.svmTrain(X_train.to_numpy(), y_train)


    # testing the model
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)
    # print('Training Accuracy SVM: %.5f'%svmPerformance(y_train, y_train_predicted))


    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)
    # print('Test Accuracy SVM: %.5f'%svmPerformance(y_test, y_test_predicted))
    return svmPerformance(y_test, y_test_predicted)

def lr_crossValidate(trainingSet, testSet):
    import warnings
    warnings.filterwarnings("ignore")
    y_train = trainingSet['decision']
    X_train = trainingSet.drop(['decision'], axis=1)
    
    """ # Trying out with simple data to see that the model actually works
    X_train = [[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1],
     [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0],
     [1,1,0,1], [1,1,1,0], [1,1,1,1]]
    # if b,d true
    y_train = [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1]
    y_train = pd.Series( (v for v in y_train) ) 
    # Output came to 1.00 accuracy on train set without normalization
    """
   
    #######################################################################
    # CONSIDER SHUFFLING DATASET EVERY EPOCH ##############################
    # RESET ALL C WITH LAMBDA AND FLIP THE SIGNS ##########################
    #######################################################################
    LR = LogisticRegression(learningRate = 0.01, numIterations = 500, penalty = 'L2', lambdaValue= 100)  
    LR.train(X_train, y_train, tol = 10 ** -6)

    y_test = testSet['decision']
    X_test = testSet.drop(['decision'], axis=1)

    # predictions, probs = LR.predict(X_train, 0.5)
    # performance = lrPerformance(predictions, y_train)
    # print('Training Accuracy LR: %.5f'%performance)

    predictions, probs = LR.predict(X_test, 0.5)
    performance = lrPerformance(predictions, y_test)
    # print('Test Accuracy LR: %.5f'%performance)
    return lrPerformance(predictions, y_test)

def lrPerformance(prediction, actual):
    diff = np.subtract(prediction, actual)
    return 1- np.sum(np.abs(diff))/len(prediction)

def svmPerformance(prediction, actual):
    prediction = np.where(prediction<0.5, 0, 1)
    actual = np.where(actual<0.5, 0, 1)

    diff = np.subtract(prediction, actual)
    return 1- np.sum(np.abs(diff))/len(prediction)

def main(trainingDataFilename, testDataFilename, modelIdx):
    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)
    y_train = trainingSet['decision']
    X_train = trainingSet.drop(['decision'], axis=1)
    
    
    """ from sklearn.linear_model import LogisticRegression
    logisticReg = LogisticRegression(random_state=0).fit(X_train, y_train)
    result = logisticReg.predict(X_train)
    print('The LogReg accuracy was:', svmPerformance(result, y_train))

    from sklearn import svm
    clf = svm.SVC()
    clf.fit(X_train, y_train) """
    
    y_test = testSet['decision']
    X_test = testSet.drop(['decision'], axis=1)

    if modelIdx == '1':
        lr(trainingSet, testSet)
    else:
        svm(trainingSet, testSet)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])