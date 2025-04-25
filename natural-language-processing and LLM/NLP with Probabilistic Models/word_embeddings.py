import re
import nltk
from nltk.tokenize import word_tokenize
import emoji
import numpy as np

from utils2 import get_dict

nltk.download('punkt')  # download pre-trained Punkt tokenizer for English

# clean and tokenize data

def tokenize(corpus):
    # replace all interrupting punctuation signs — such as commas and exclamation marks — with periods
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)  # tokenize string to words
    # get rid of numbers and punctuation other than periods, and convert all the remaining tokens to lowercase
    data = [ ch.lower() for ch in data
             if ch.isalpha()
             or ch == '.'
             or emoji.get_emoji_regexp().search(ch)
           ]
    return data

# extract context and center words

def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1

# Transform words into numerical vectors

# To create one-hot word vectors, you can start by mapping each unique word to a unique integer
words = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning']
word2Ind, Ind2word = get_dict(words)

def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1 
    return one_hot_vector

def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)    
    return context_words_vectors

# build a training set for the CBOW model

def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

# The continuous bag-of-words model

def relu(z):
    result = z.copy()
    result[result < 0] = 0
    return result

def softmax(z):
    e_z = np.exp(z)
    sum_e_z = np.sum(e_z)
    return e_z / sum_e_z

#Set 𝑁 equal to 3. Remember that 𝑁 is a hyperparameter of the CBOW model that represents the size of the word embedding vectors, as well as the size of the hidden layer.
N = 3

W1 = np.array([[ 0.41687358,  0.08854191, -0.23495225,  0.28320538,  0.41800106],
               [ 0.32735501,  0.22795148, -0.23951958,  0.4117634 , -0.23924344],
               [ 0.26637602, -0.23846886, -0.37770863, -0.11399446,  0.34008124]])
W2 = np.array([[-0.22182064, -0.43008631,  0.13310965],
               [ 0.08476603,  0.08123194,  0.1772054 ],
               [ 0.1871551 , -0.06107263, -0.1790735 ],
               [ 0.07055222, -0.02015138,  0.36107434],
               [ 0.33480474, -0.39423389, -0.43959196]])
b1 = np.array([[ 0.09688219],
               [ 0.29239497],
               [-0.27364426]])
b2 = np.array([[ 0.0352008 ],
               [-0.36393384],
               [-0.12775555],
               [-0.34802326],
               [-0.07017815]])

training_examples = get_training_example(words, 2, word2Ind, V)
x_array, y_array = next(training_examples)
#Now convert these vectors into matrices (or 2D arrays) to be able to perform matrix multiplication
x = x_array.copy() # (5,)
x.shape = (x_array.shape[0], 1) # (5, 1)
y = y_array.copy() # (5,)
y.shape = (y_array.shape[0], 1) # (5, 1)

# forward propagation
z1 = np.dot(W1, x) + b1
h = relu(z1)
z2 = np.dot(W2, h) + b2
y_hat = softmax(z2)

# Cost
def cross_entropy_loss(y_predicted, y_actual):
    loss = np.sum(-np.log(y_predicted)*y_actual)
    return loss

cross_entropy_loss(y_hat, y)

# Backpropagation
grad_b2 = y_hat - y
grad_W2 = np.dot(y_hat - y, h.T)
grad_b1 = relu(np.dot(W2.T, y_hat - y))
grad_W1 = np.dot(relu(np.dot(W2.T, y_hat - y)), x.T)

#Gradient descend
alpha = 0.03
W1_new = W1 - alpha * grad_W1
W2_new = W2 - alpha * grad_W2
b1_new = b1 - alpha * grad_b1
b2_new = b2 - alpha * grad_b2

# Extracting word embedding vectors

#Method one W1 columns
for word in word2Ind:
    word_embedding_vector = W1[:, word2Ind[word]]
    print(f'{word}: {word_embedding_vector}')

#Method two W2 rows
for word in word2Ind:
    word_embedding_vector = W2.T[:, word2Ind[word]]
    print(f'{word}: {word_embedding_vector}')

#Method three W1 columns and W2 rows average
W3 = (W1+W2.T)/2
for word in word2Ind:
    word_embedding_vector = W3[:, word2Ind[word]]
    print(f'{word}: {word_embedding_vector}')


