import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import load_tweets, process_tweet

import w1_unittest

# Load positive and negative tweets
all_positive_tweets, all_negative_tweets = load_tweets()
# View
print(f"The number of positive tweets: {len(all_positive_tweets)}")
print(f"The number of negative tweets: {len(all_negative_tweets)}")
tweet_number = 4
print('Positive tweet example:')
print(all_positive_tweets[tweet_number])
print('\nNegative tweet example:')

# Process all the tweets: tokenize the string, remove tickers, handles, punctuation and stopwords, stem the words
all_positive_tweets_processed = [process_tweet(tweet) for tweet in all_positive_tweets]
all_negative_tweets_processed = [process_tweet(tweet) for tweet in all_negative_tweets]
# View
tweet_number = 4
print('Positive processed tweet example:')
print(all_positive_tweets_processed[tweet_number])
print('\nNegative processed tweet example:')
print(all_negative_tweets_processed[tweet_number])

# Split training sets

# Split positive set into validation and training
val_pos = all_positive_tweets_processed[4000:]
train_pos = all_positive_tweets_processed[:4000]
# Split negative set into validation and training
val_neg = all_negative_tweets_processed[4000:]
train_neg = all_negative_tweets_processed[:4000]

train_x = train_pos + train_neg
val_x  = val_pos + val_neg

# Set the labels for the training and validation set (1 for positive, 0 for negative)
train_y = [[1] for _ in train_pos] + [[0] for _ in train_neg]
val_y  = [[1] for _ in val_pos] + [[0] for _ in val_neg]

#View
print(f"There are {len(train_x)} sentences for training.")
print(f"There are {len(train_y)} labels for training.\n")
print(f"There are {len(val_x)} sentences for validation.")
print(f"There are {len(val_y)} labels for validation.")

# Create a vocabulary by assigning an index to every word in training set plus the special tokens '' (padding) and '[UNK]' (words not in vocabulary)

# GRADED FUNCTION: build_vocabulary
def build_vocabulary(corpus):
    '''Function that builds a vocabulary from the given corpus
    Input: 
        - corpus (list): the corpus
    Output:
        - vocab (dict): Dictionary of all the words in the corpus.
                The keys are the words and the values are integers.
    '''
    # The vocabulary includes special tokens like padding token and token for unknown words
    # Keys are words and values are distinct integers (increasing by one from 0)
    vocab = {'': 0, '[UNK]': 1} 
    ### START CODE HERE ###
    # For each tweet in the training set
    for tweet in corpus:
        # For each word in the tweet
        for word in tweet:
            # If the word is not in vocabulary yet, add it to vocabulary
            if word not in vocab:
                vocab[word] = len(vocab)
    ### END CODE HERE ###
    return vocab

vocab = build_vocabulary(train_x)
num_words = len(vocab)
print(f"Vocabulary contains {num_words} words\n")
print(vocab)

# Test the build_vocabulary function
w1_unittest.test_build_vocabulary(build_vocabulary)

# Next, you will write a function that will convert each tweet to a tensor (a list of integer IDs representing the processed tweet).
# You will also use padding if necessary.

# GRADED FUNCTION: max_length
def max_length(training_x, validation_x):
    """Computes the length of the longest tweet in the training and validation sets.

    Args:
        training_x (list): The tweets in the training set.
        validation_x (list): The tweets in the validation set.

    Returns:
        int: Length of the longest tweet.
    """
    ### START CODE HERE ###
    max_len = 0
    for tweet in training_x + validation_x:
        if len(tweet) > max_len:
            max_len = len(tweet)
    ### END CODE HERE ###
    return max_len

max_len = max_length(train_x, val_x)
print(f'The length of the longest tweet is {max_len} tokens.')

# Test your max_len function
w1_unittest.test_max_length(max_length)

# GRADED FUNCTION: padded_sequence
def padded_sequence(tweet, vocab_dict, max_len, unk_token='[UNK]'):
    """transform sequences of words into padded sequences of numbers

    Args:
        tweet (list): A single tweet encoded as a list of strings.
        vocab_dict (dict): Vocabulary.
        max_len (int): Length of the longest tweet.
        unk_token (str, optional): Unknown token. Defaults to '[UNK]'.

    Returns:
        list: Padded tweet encoded as a list of int.
    """
    ### START CODE HERE ### 
    # Find the ID of the UNK token, to use it when you encounter a new word
    unk_ID = vocab_dict[unk_token] 
    # First convert the words to integers by looking up the vocab_dict
    padded_tensor = []
    for word in tweet:
        if word in vocab_dict:
            padded_tensor.append(vocab_dict[word])
        else:
            padded_tensor.append(unk_ID)
    # Then pad the tensor with zeroes up to the length max_len
    while len(padded_tensor) < max_len:
        padded_tensor.append(0)
    ### END CODE HERE ###
    return padded_tensor

# Test your padded_sequence function
w1_unittest.test_padded_sequence(padded_sequence)

train_x_padded = [padded_sequence(x, vocab, max_len) for x in train_x]
val_x_padded = [padded_sequence(x, vocab, max_len) for x in val_x]

# Define the structure of the neural network layers

# GRADED FUNCTION: relu
def relu(x):
    '''Relu activation function implementation
    Input:
        - x (numpy array)
    Output:
        - activation (numpy array): input with negative values set to zero
    '''
    ### START CODE HERE ###
    activation = np.maximum(0, x)
    ### END CODE HERE ###
    return activation

# Test your relu function
w1_unittest.test_relu(relu)

# GRADED FUNCTION: sigmoid
def sigmoid(x):
    '''Sigmoid activation function implementation
    Input:
        - x (numpy array)
    Output:
        - activation (numpy array)
    '''
    ### START CODE HERE ###
    activation = 1 / (1 + np.exp(-x))
    ### END CODE HERE ###
    return activation

# Test your sigmoid function
w1_unittest.test_sigmoid(sigmoid)

# Handle a Dense layer via a class

# GRADED CLASS: Dense
class Dense():
    """
    A dense (fully-connected) layer.
    """
    # Please implement '__init__'
    def __init__(self, n_units, input_shape, activation, stdev=0.1, random_seed=42): 
        # Set the number of units in this layer
        self.n_units = n_units
        # Set the random key for initializing weights
        self.random_generator = np.random.default_rng(seed=random_seed)
        self.activation = activation
        ### START CODE HERE ###
        # Generate the weight matrix from a normal distribution and standard deviation of 'stdev'
        # Set the size of the matrix w
        w = self.random_generator.normal(scale=stdev, size = (input_shape[1], n_units))
        ### END CODE HERE ##
        self.weights = w

    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        ### START CODE HERE ###
        # Matrix multiply x and the weight matrix
        dense = np.dot(x, self.weights)
        # Apply the activation function
        dense = self.activation(dense)
        ### END CODE HERE ###
        return dense

# Test your Dense class
w1_unittest.test_Dense(Dense)

# Create neural network model for classification using TensorFlow

# GRADED FUNCTION: create_model
def create_model(num_words, embedding_dim, max_len):
    """
    Creates a text classifier model
    
    Args:
        num_words (int): size of the vocabulary for the Embedding layer input
        embedding_dim (int): dimensionality of the Embedding layer output
        max_len (int): length of the input sequences
    
    Returns:
        model (tf.keras Model): the text classifier model
    """ 
    tf.random.set_seed(123)
    ### START CODE HERE
    model = tf.keras.Sequential([ 
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, 'sigmoid')
    ])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    ### END CODE HERE
    return model

# Create the model
model = create_model(num_words=num_words, embedding_dim=16, max_len=max_len)

# Test your create_model function
w1_unittest.test_model(create_model)

# Prepare the data
train_x_prepared = np.array(train_x_padded)
val_x_prepared = np.array(val_x_padded)

train_y_prepared = np.array(train_y)
val_y_prepared = np.array(val_y)

# Fit the model
history = model.fit(train_x_prepared, train_y_prepared, epochs=20, validation_data=(val_x_prepared, val_y_prepared))

# Predict

# Prepare an example with 10 positive and 10 negative tweets.
example_for_prediction = np.append(val_x_prepared[0:10], val_x_prepared[-10:], axis=0)
# Make a prediction on the tweets.
model.predict(example_for_prediction)

def get_prediction_from_tweet(tweet, model, vocab, max_len):
    tweet = process_tweet(tweet)
    tweet = padded_sequence(tweet, vocab, max_len)
    tweet = np.array([tweet])
    prediction = model.predict(tweet, verbose=False)
    return prediction[0][0]

unseen_tweet = '@DLAI @NLP_team_dlai OMG!!! what a daaay, wow, wow. This AsSiGnMeNt was gr8.'
prediction_unseen = get_prediction_from_tweet(unseen_tweet, model, vocab, max_len)
print(f"Model prediction on unseen tweet: {prediction_unseen}")

# Extract the Word Embeddings

# Get the embedding layer
embeddings_layer = model.layers[0]
# Get the weights of the embedding layer
embeddings = embeddings_layer.get_weights()[0]
print(f"Weights of embedding layer have shape: {embeddings.shape}")

# Visualize

# PCA with two dimensions
pca = PCA(n_components=2)
# Dimensionality reduction of the word embeddings
embeddings_2D = pca.fit_transform(embeddings)
# Reducing data to two dimensions using the dimensionality reduction algorithm PCA allows for plotting on a graph and thus visualization

#Selection of negative and positive words
neg_words = ['bad', 'hurt', 'sad', 'hate', 'worst']
pos_words = ['best', 'good', 'nice', 'love', 'better', ':)']
#Index of each selected word
neg_n = [vocab[w] for w in neg_words]
pos_n = [vocab[w] for w in pos_words]

plt.figure()
#Scatter plot for negative words
plt.scatter(embeddings_2D[neg_n][:,0], embeddings_2D[neg_n][:,1], color = 'r')
for i, txt in enumerate(neg_words):
    plt.annotate(txt, (embeddings_2D[neg_n][i,0], embeddings_2D[neg_n][i,1]))
#Scatter plot for positive words
plt.scatter(embeddings_2D[pos_n][:,0], embeddings_2D[pos_n][:,1], color = 'g')
for i, txt in enumerate(pos_words):
    plt.annotate(txt,(embeddings_2D[pos_n][i,0], embeddings_2D[pos_n][i,1]))
plt.title('Word embeddings in 2d')
plt.show()
