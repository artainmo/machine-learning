import numpy as np

def softmax(predicted):
    return np.exp(predicted) / np.sum(np.exp(predicted))

#Output of total vector values equals one
def derivative_softmax(x):
    return x * (1 - x)

#activation function, sets value between 0 and 1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):#used to find gradient, calculates slope of predicted value
    return x * (1 - x)

#activation function sets value between -1,1
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative_tanh(x):
    return 1 - np.square(x)

#Value stays same unless under zero than equal to zero
def relu(x):
    return max(0, x)

def call_relu(x):
    return np.array([[relu(elem) for elem in x[0]]], dtype=np.float128)

def derivative_relu(x):
    if x <= 0:
        return 0
    else:
        return 1

def call_derivative_relu(x):
    return np.array([[relu(elem) for elem in x[0]]], dtype=np.float128)
