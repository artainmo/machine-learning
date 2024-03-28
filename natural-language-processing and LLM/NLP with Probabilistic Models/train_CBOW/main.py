import numpy as np
from utils2 import get_dict

# Define the size of the word embedding vectors and hidden layer
N = 3

# Define vocabulary size
V = 5

# Before you start training the neural network, you need to initialize the weight matrices and bias vectors with random values.
W1 = np.array([[ 0.41687358,  0.08854191, -0.23495225,  0.28320538,  0.41800106],
               [ 0.32735501,  0.22795148, -0.23951958,  0.4117634 , -0.23924344],
               [ 0.26637602, -0.23846886, -0.37770863, -0.11399446,  0.34008124]])

# Define second matrix of weights
W2 = np.array([[-0.22182064, -0.43008631,  0.13310965],
               [ 0.08476603,  0.08123194,  0.1772054 ],
               [ 0.1871551 , -0.06107263, -0.1790735 ],
               [ 0.07055222, -0.02015138,  0.36107434],
               [ 0.33480474, -0.39423389, -0.43959196]])

# Define first vector of biases
b1 = np.array([[ 0.09688219],
               [ 0.29239497],
               [-0.27364426]])

# Define second vector of biases
b2 = np.array([[ 0.0352008 ],
               [-0.36393384],
               [-0.12775555],
               [-0.34802326],
               [-0.07017815]])

print(f'V (vocabulary size): {V}')
print(f'N (embedding size / size of the hidden layer): {N}')
print(f'size of W1: {W1.shape} (NxV)')
print(f'size of b1: {b1.shape} (Nx1)')
print(f'size of W2: {W2.shape} (VxN)')
print(f'size of b2: {b2.shape} (Vx1)')
# Output:
# V (vocabulary size): 5
# N (embedding size / size of the hidden layer): 3
# size of W1: (3, 5) (NxV)
# size of b1: (3, 1) (Nx1)
# size of W2: (5, 3) (VxN)
# size of b2: (5, 1) (Vx1)

# Define the tokenized version of the corpus
words = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning']

# Get 'word2Ind' and 'Ind2word' dictionaries for the tokenized corpus
word2Ind, Ind2word = get_dict(words)

# Define the 'get_windows' function as seen in a previous notebook
def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1

# Define the 'word_to_one_hot_vector' function as seen in a previous notebook
def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector

# Define the 'context_words_to_vector' function as seen in a previous notebook
def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors

# Define the generator function 'get_training_example' as seen in a previous notebook
def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

# Save generator object in the 'training_examples' variable with the desired arguments
training_examples = get_training_example(words, 2, word2Ind, V)
# Get first values from generator
x_array, y_array = next(training_examples)

# convert these vectors into matrices (or 2D arrays) to be able to perform matrix multiplication
# Copy vector
x = x_array.copy()
# Reshape it
x.shape = (V, 1)
# Print it
print(f'x:\n{x}\n')
# Copy vector
y = y_array.copy()
# Reshape it
y.shape = (V, 1)
# Print it
print(f'y:\n{y}')

# Define the 'relu' function as seen in the previous lecture notebook
def relu(z):
    result = z.copy()
    result[result < 0] = 0
    return result

# Define the 'softmax' function as seen in the previous lecture notebook
def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z / sum_e_z

# Forward propagation

# Compute z1 (values of first hidden layer before applying the ReLU function)
z1 = np.dot(W1, x) + b1
# Compute h (z1 after applying ReLU function)
h = relu(z1)
# Compute z2 (values of the output layer before applying the softmax function)
z2 = np.dot(W2, h) + b2
# Compute y_hat (z2 after applying softmax function)
y_hat = softmax(z2)

# As you've performed the calculations with random matrices and vectors (apart from the input vector), the output of the neural network is essentially random at this point. The learning process will adjust the weights and biases to match the actual targets better.

# Now that you have the network's prediction, you can calculate the cross-entropy loss to determine how accurate the prediction was compared to the actual target.

def cross_entropy_loss(y_predicted, y_actual):
    loss = np.sum(np.log(y_predicted) * y_actual) * -1
    return loss

cross_entropy_loss(y_hat, y)

# Backward propagation

# Compute vector with partial derivatives of loss function with respect to b2
grad_b2 = y_hat - y
# Compute matrix with partial derivatives of loss function with respect to W2
grad_W2 = np.dot(y_hat - y, h.T)
# Compute vector with partial derivatives of loss function with respect to b1
grad_b1 = relu(np.dot(W2.T, y_hat - y))
# Compute matrix with partial derivatives of loss function with respect to W1
grad_W1 = np.dot(relu(np.dot(W2.T, y_hat - y)), x.T)


#During the gradient descent phase, you will update the weights and biases by subtracting 𝛼 times the gradient from the original matrices and vectors.

# Define alpha
alpha = 0.03

# Compute updated W1
W1_new = W1 - alpha * grad_W1
# Compute updated W2
W2_new = W2 - alpha * grad_W2
# Compute updated b1
b1_new = b1 - alpha * grad_b1
# Compute updated b2
b2_new = b2 - alpha * grad_b2
