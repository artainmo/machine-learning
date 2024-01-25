import numpy as np

#Return vector of output node errors, if you want one cost sum them together
def mean_square_error(predicted, expected):
    if predicted.ndim == 1:
        predicted = np.array([predicted])
    return np.sum(np.square(predicted - expected)) / predicted.shape[1]

def derivative_mean_square_error(predicted, expected):
    return predicted - expected

#Also called logistic loss function
def cross_entropy(predicted, expected):
    if predicted.ndim == 1:
        predicted = np.array([predicted])
    return np.sum((-expected * np.log(1e-15 + predicted)) - ((1 - expected) * np.log(1e-15 + 1 - predicted))) / predicted.shape[1] #1e-15 is used to never do log of 0 which is equal to inf
    # return np.sum((-expected * np.log(predicted)) - ((1 - expected) * np.log(1 - predicted))) / predicted.shape[1]


def derivative_cross_entropy(predicted, expected):
    return predicted - expected
