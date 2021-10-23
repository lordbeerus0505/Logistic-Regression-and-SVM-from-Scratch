import numpy as np
import pandas as pd
from numpy.linalg import norm
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

def calculate_cost_gradient(W, X_batch, Y_batch):
    yi_hat = np.dot(X_batch, W)
    product = yi_hat*Y_batch
    
    netResult = np.where(product<1,Y_batch,0) 
    # when mistake the product is less than 1.
    # since 0 into anything anyway 0 can put it here and then multiply later
    netResult = np.tile(netResult.transpose(),(X_batch.shape[1],1))
    delta_ji = netResult.T*X_batch
    
    delta = 1/regularization_strength*W - delta_ji
    delta = np.sum(delta, axis=0)/len(Y_batch)
    return delta


def gradientDescent(X, Y):
    #Change all X,Y to xtrain y train
    max_epochs = 500
    weights = np.zeros(X.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.001  # in percent
    

    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        print("Epoch %s completed"%epoch)
        descent = calculate_cost_gradient(weights, X, Y)
        # weights = weights - (learning_rate * descent)
        delta_w = learning_rate * descent
        # import pdb; pdb.set_trace()
        if norm(delta_w) >= tol: 
            weights -= delta_w
        else:
            break
    return weights


# def init():
#     print("reading dataset...")

#     trainingSet = pd.read_csv('trainingSet.csv', dtype=np.float64)
#     testSet = pd.read_csv('testSet.csv', dtype=np.float64)
#     Y = trainingSet.loc[:, 'decision']
#     X = trainingSet.drop(['decision'], axis=1)
#     Y= np.where(Y==0,-1,Y)
    

#     # insert 1 in every row for intercept b
#     X.insert(loc=len(X.columns), column='intercept', value=1)
    
#     X_train = X[:]
#     y_train = Y[:]
#     y_test = testSet.loc[:, 'decision']
#     y_test= np.where(y_test==0,-1,y_test)
#     X_test = testSet.drop(['decision'], axis=1)

#     X_test.insert(loc=len(X_test.columns), column='intercept', value=1)

#     W = gradientDescent(X_train.to_numpy(), y_train)


#     # testing the model
#     print("testing the model...")
#     y_train_predicted = np.array([])
#     # import pdb; pdb.set_trace()
#     for i in range(X_train.shape[0]):
#         yp = np.sign(np.dot(X_train.to_numpy()[i], W))
#         y_train_predicted = np.append(y_train_predicted, yp)

#     print("accuracy on train dataset: {}".format(accuracy_score(y_train, y_train_predicted)))

#     y_test_predicted = np.array([])
#     for i in range(X_test.shape[0]):
#         yp = np.sign(np.dot(X_test.to_numpy()[i], W))
#         y_test_predicted = np.append(y_test_predicted, yp)

#     print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))


# set hyper-parameters and call init
# regularization_strength = 100
# learning_rate = 0.01
# tol = 10**-2


class SupportVectorMachine(object):
    def __init__(self, learning_rate = 0.5, numIterations = 500, penalty = None, C = 100, tol = 10**-6):
        self.learningRate = learning_rate
        self.numIterations = numIterations
        self.penalty = penalty
        self.C = C
        self.tol = tol
    
    def svmGradientCost(self, W, X_batch, Y_batch):
        yi_hat = np.dot(X_batch, W)
        product = yi_hat*Y_batch
        
        netResult = np.where(product<1,Y_batch,0) 
        # when mistake the product is less than 1.
        # since 0 into anything anyway 0 can put it here and then multiply later
        netResult = np.tile(netResult.transpose(),(X_batch.shape[1],1))
        delta_ji = netResult.T*X_batch
        
        delta = 1/self.C*W - delta_ji
        delta = np.sum(delta, axis=0)/len(Y_batch)
        return delta


    def svmTrain(self, X, Y):
        #Change all X,Y to xtrain y train
        max_epochs = 500
        weights = np.zeros(X.shape[1])
        nth = 0
        prev_cost = float("inf")
        cost_threshold = 0.001  # in percent
        svmObj = SupportVectorMachine()

        # Gradient descent
        for epoch in range(self.numIterations):
            # print("Epoch %s completed"%epoch)
            descent = svmObj.svmGradientCost(weights, X, Y)
            delta_w = self.learningRate * descent

            if norm(delta_w) >= self.tol: 
                weights -= delta_w
            else:
                break
        return weights

def svmPerformance(prediction, actual):
    prediction = np.where(prediction<0.5, 0, 1)
    actual = np.where(actual<0.5, 0, 1)

    diff = np.subtract(prediction, actual)
    return 1- np.sum(np.abs(diff))/len(prediction)

def init():

    trainingSet = pd.read_csv('trainingSet.csv', dtype=np.float64)
    testSet = pd.read_csv('testSet.csv', dtype=np.float64)
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
    svm = SupportVectorMachine(learning_rate = 0.5, numIterations = 500, penalty = 'L2', C = 100)
    W = svm.svmTrain(X_train.to_numpy(), y_train)


    # testing the model
    print("Using the class object testing the model...")
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)
    # import pdb; pdb.set_trace()
    print("accuracy on train dataset: {}".format(svmPerformance(y_train, y_train_predicted)))

    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)

    print("accuracy on test dataset: {}".format(svmPerformance(y_test, y_test_predicted)))

init() 