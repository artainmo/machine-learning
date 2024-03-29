# In this lecture notebook you will be given an introduction to the continuous bag-of-words model, its activation functions and some considerations when working with Numpy.

import numpy as np

# Relu leaves positive numbers as they are and transform negative numbers into 0
def relu(z):
    result = z.copy()
    result[result < 0] = 0
    return result

# Softmax returns a probability that lies between 0 and 1
def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z, axis=0)
    return e_z / sum_e_z


