import numpy as np
from .MyStats import *

#Softmax should be used on at least two output nodes, this transforms the y from size one to two
def softmax_compatible(y):
    new = np.zeros((y.shape[0], 2))
    for i in range(y.shape[0]):
        if y[i] == 1:
            new[i] = np.append(y[i], np.array([0], dtype=np.float128))
        elif y[i] == 0:
            new[i] = np.append(y[i], np.array([1], dtype=np.float128))
    return new

def zscore(x):
    return (x - mean(x)) / standard_deviation(x)

def normalization_zscore(x):
    if x.ndim == 1:
        x = np.array([x])
        x = x.T
    return np.array([zscore(column) for column in x.T], dtype=np.float128).T

def minmax_normalization(x_values):
    if x_values.ndim == 1:
        x_values = np.array([x_values])
        x_values = x_values.T
    i = 0
    x_values = x_values.transpose()
    while i < x_values.shape[0]:
        max_ = max(x_values[i])
        min_ = min(x_values[i])
        range = max_ - min_
        x_values[i] = np.divide(np.subtract(x_values[i], min_), range)
        i += 1
    return x_values.transpose()

def get_train_test(x, y, proportion=0.8):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return (training_set[:,:-y.shape[1]], training_set[:,-y.shape[1]:], test_set[:,:-y.shape[1]], test_set[:,-y.shape[1]:]) #(train_x, train_y, test_x, test_y)


#Set highest value to one and rest to zero
def softmax_to_answer2(predicted):
    _max = max(predicted)
    for i in range(predicted.shape[0]):
        if predicted[i] == _max:
            predicted[i] = 1
        else:
            predicted[i] = 0
    return predicted

def softmax_to_answer(y):
    return np.array([softmax_to_answer2(predicted) for predicted in y], dtype=np.float128)

#If value higher than division_point (default 0.5), set to one otherwise set to 0
def sigmoid_to_answer(predicted, division_point):
    for i in range(predicted.shape[0]):
        if predicted[i] > division_point:
            predicted[i] = 1
        else:
            predicted[i] = 0
    return predicted

def sigmoid_to_answer(y, division_point=0.5):
    return np.array([softmax_to_answer2(predicted) for predicted in y], dtype=np.float128)


def relu_to_answer(y):
    return y
