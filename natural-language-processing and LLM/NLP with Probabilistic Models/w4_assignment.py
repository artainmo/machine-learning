# In this assignment, you will practice how to compute word embeddings and use them for sentiment analysis.

# Import Python libraries and helper functions (in utils2)
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils2 import sigmoid, get_batches, compute_pca, get_dict
import w4_unittest

nltk.download('punkt')

# Download sentence tokenizer
nltk.data.path.append('.')

# Load, tokenize and process the data
import re                                                           #  Load the Regex-modul
with open('./data/shakespeare.txt') as f:
    data = f.read()                                                 #  Read in the data
data = re.sub(r'[,!?;-]', '.',data)                                 #  Punctuations are replaced by .
data = nltk.word_tokenize(data)                                     #  Tokenize string to words
data = [ ch.lower() for ch in data if ch.isalpha() or ch == '.']    #  Lower case and drop non-alphabetical tokens
print("Number of tokens:", len(data),'\n', data[:15])               #  print data sample

# get_dict creates two dictionaries, converting words to indices and viceversa.
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)

# Train the model

# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C1 GRADED FUNCTION: initialize_model
def initialize_model(N, V, random_seed=1):
    '''
    Inputs:
        N:  dimension of hidden vector
        V:  dimension of vocabulary
        random_seed: random seed for consistent results in the unit tests
     Outputs:
        W1, W2, b1, b2: initialized weights and biases
    '''
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    np.random.seed(random_seed)
    # W1 has shape (N,V)
    W1 = np.random.rand(N, V)
    # W2 has shape (V,N)
    W2 = np.random.rand(V, N)
    # b1 has shape (N,1)
    b1 = np.random.rand(N, 1)
    # b2 has shape (V,1)
    b2 = np.random.rand(V, 1)
    ### END CODE HERE ###
    return W1, W2, b1, b2

# Test your function
w4_unittest.test_initialize_model(initialize_model)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C2 GRADED FUNCTION: softmax
def softmax(z):
    '''
    Inputs:
        z: output scores from the hidden layer
    Outputs:
        yhat: prediction (estimate of y)
    '''
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate yhat (softmax)
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z, axis=0)
    yhat = e_z / sum_e_z
    ### END CODE HERE ###
    return yhat

# Test your function
w4_unittest.test_softmax(softmax)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C3 GRADED FUNCTION: forward_prop
def forward_prop(x, W1, W2, b1, b2):
    '''
    Inputs: 
        x:  average one hot vector for the context 
        W1, W2, b1, b2:  matrices and biases to be learned
     Outputs: 
        z:  output score vector
    '''
    
    ### START CODE HERE (Replace instances of 'None' with your own code) ###
    # Calculate h
    h = np.dot(W1, x) + b1
    # Apply the relu on h, 
    # store the relu in h
    h = np.maximum(0, h)
    # Calculate z
    z = np.dot(W2, h) + b2
    ### END CODE HERE ###
    return z, h

# Test your function
w4_unittest.test_forward_prop(forward_prop)

# compute_cost: cross-entropy cost function
def compute_cost(y, yhat, batch_size):
    logprobs = np.multiply(np.log(yhat), y)
    cost = - 1/batch_size * np.sum(logprobs)
    # squeeze is used to remove single-dimensional entries from the shape of an array
    # for example shape (1, 2, 3) becomes (2, 3)
    cost = np.squeeze(cost)
    return cost

# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C4 GRADED FUNCTION: back_prop
def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    '''
    Inputs: 
        x:  average one hot vector for the context 
        yhat: prediction (estimate of y)
        y:  target vector
        h:  hidden vector (see eq. 1)
        W1, W2, b1, b2:  matrices and biases  
        batch_size: batch size 
     Outputs: 
        grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases   
    '''
    # Compute z1 as "W1⋅x + b1"
    z1 = np.dot(W1, x) + b1
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    # Compute l1 as W2^T (Yhat - Y)
    l1 = np.dot(W2.T, yhat - y)
    # if z1 < 0, then l1 = 0 
    # otherwise l1 = l1
    # (this is already implemented for you and is same as relu activation function)
    l1[z1 < 0] = 0 # use "l1" to compute gradients below
    # compute the gradient for W1
    grad_W1 = np.dot(l1, x.T) / batch_size
    # Compute gradient of W2
    grad_W2 = np.dot(yhat - y, h.T) / batch_size
    # compute gradient for b1
    grad_b1 = np.sum(l1, axis=1, keepdims=True) / batch_size
    # compute gradient for b2
    grad_b2 = np.sum(yhat - y, axis=1, keepdims=True) / batch_size
    ### END CODE HERE #### 
    return grad_W1, grad_W2, grad_b1, grad_b2

# Test your function
w4_unittest.test_back_prop(back_prop)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
# UNQ_C5 GRADED FUNCTION: gradient_descent
def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03, 
                     random_seed=282, initialize_model=initialize_model, 
                     get_batches=get_batches, forward_prop=forward_prop, 
                     softmax=softmax, compute_cost=compute_cost, 
                     back_prop=back_prop):
    '''
    This is the gradient_descent function
      Inputs: 
        data:      text
        word2Ind:  words to Indices
        N:         dimension of hidden vector  
        V:         dimension of vocabulary 
        num_iters: number of iterations  
        random_seed: random seed to initialize the model's matrices and vectors
        initialize_model: your implementation of the function to initialize the model
        get_batches: function to get the data in batches
        forward_prop: your implementation of the function to perform forward propagation
        softmax: your implementation of the softmax function
        compute_cost: cost function (Cross entropy)
        back_prop: your implementation of the function to perform backward propagation
     Outputs: 
        W1, W2, b1, b2:  updated matrices and biases after num_iters iterations
    '''
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=random_seed) #W1=(N,V) and W2=(V,N)
    batch_size = 128
#    batch_size = 512
    iters = 0
    C = 2
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        ### START CODE HERE (Replace instances of 'None' with your own code) ###                
        # get z and h
        z, h = forward_prop(x, W1, W2, b1, b2)
        # get yhat
        yhat = softmax(z)
        # get cost
        cost = compute_cost(y, yhat, batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")    
        # get gradients
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        # update weights and biases
        W1 = W1 - (alpha * grad_W1)
        W2 = W2 - (alpha * grad_W2)
        b1 = b1 - (alpha * grad_b1)
        b2 = b2 - (alpha * grad_b2)
        ### END CODE HERE ###
        iters +=1 
        if iters == num_iters: 
            break
        if iters % 100 == 0:
            alpha *= 0.66
    return W1, W2, b1, b2

# Test your function
w4_unittest.test_gradient_descent(gradient_descent, data, word2Ind, N=10, V=len(word2Ind), num_iters=15)

C = 2
N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
num_iters = 150
print("Call gradient_descent")
W1, W2, b1, b2 = gradient_descent(data, word2Ind, N, V, num_iters)

# Visualize the word vectors

from matplotlib import pyplot
%config InlineBackend.figure_format = 'svg'
words = ['king', 'queen','lord','man', 'woman','dog','wolf',
         'rich','happy','sad']

embs = (W1.T + W2)/2.0 # Get word embeddings using third extraction method

# given a list of words and the embeddings, it returns a matrix with all the embeddings
idx = [word2Ind[word] for word in words]
X = embs[idx, :] # selecting all columns from the rows specified by 'idx'

result = compute_pca(X, 2) #lower dimensionality to enable display on 2D graph
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

# In graph similar words should be close to each other if word embedding is good
