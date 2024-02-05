import pdb
import pickle
import string

import time

import nltk
import numpy as np
from nltk.corpus import stopwords, twitter_samples

from utils import (cosine_similarity, get_dict,
                   process_tweet)
from os import getcwd

import w4_unittest

nltk.download('stopwords')
nltk.download('twitter_samples')

# add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
filePath = f"{getcwd()}/tmp2/"
nltk.data.path.append(filePath)

# To do the assignment on the Coursera workspace, we'll use the subset of word embeddings.
en_embeddings_subset = pickle.load(open("./data/en_embeddings.p", "rb"))
fr_embeddings_subset = pickle.load(open("./data/fr_embeddings.p", "rb"))

# loading the english to french dictionaries
en_fr_train = get_dict('./data/en-fr.train.txt')
print('The length of the English to French training dictionary is', len(en_fr_train))
en_fr_test = get_dict('./data/en-fr.test.txt')
print('The length of the English to French test dictionary is', len(en_fr_test))

# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_matrices(en_fr, french_vecs, english_vecs):
    """
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    """
    ### START CODE HERE ###
    # X_l and Y_l are lists of the english and french word embeddings
    X_l = list()
    Y_l = list()
    # get the english words (the keys in the dictionary) and store in a set()
    english_set = set(english_vecs.keys())
    # get the french words (keys in the dictionary) and store in a set()
    french_set = set(french_vecs.keys())
    # store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    french_words = set(en_fr.values())
    # loop through all english, french word pairs in the english french dictionary
    for en_word, fr_word in en_fr.items():
        # check that the french word has an embedding and that the english word has an embedding
        if fr_word in french_set and en_word in english_set:
            # get the english embedding
            en_vec = english_vecs[en_word]
            # get the french embedding
            fr_vec = french_vecs[fr_word]
            # add the english embedding to the list
            X_l.append(en_vec)
            # add the french embedding to the list
            Y_l.append(fr_vec)
    # stack the vectors of X_l into a matrix X
    X = np.array(X_l)
    # stack the vectors of Y_l into a matrix Y
    Y = np.array(Y_l)
    ### END CODE HERE ###
    return X, Y

# getting the training set:
X_train, Y_train = get_matrices(en_fr_train, fr_embeddings_subset, en_embeddings_subset)
# Test your function
w4_unittest.test_get_matrices(get_matrices)
print('\033[0m', end='')

# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_loss(X, Y, R):
    '''
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
    ### START CODE HERE ###
    # m is the number of rows in X
    m = len(X)
    # diff is XR - Y    
    diff = np.dot(X, R) - Y
    # diff_squared is the element-wise square of the difference    
    diff_squared = np.power(diff, 2)
    # sum_diff_squared is the sum of the squared elements
    sum_diff_squared = np.sum(diff_squared)
    # loss i is the sum_diff_squared divided by the number of examples (m)
    loss = sum_diff_squared / m
    ### END CODE HERE ###
    return loss

# Testing your implementation.
np.random.seed(123)
m = 10
n = 5
X = np.random.rand(m, n)
Y = np.random.rand(m, n) * .1
R = np.random.rand(n, n)
print(f"Expected loss for an experiment with random matrices: {compute_loss(X, Y, R):.4f}" )
w4_unittest.test_compute_loss(compute_loss)
print('\033[0m', end='')

# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_gradient(X, Y, R):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a scalar value - gradient of the loss function L for given X, Y and R.
    '''
    ### START CODE HERE ###
    # m is the number of rows in X
    m = len(X)
    # gradient is X^T(XR - Y) * 2/m
    gradient = np.dot(X.T, (np.dot(X, R) - Y)) * (2/m)
    ### END CODE HERE ###
    return gradient

# Testing your implementation.
np.random.seed(123)
m = 10
n = 5
X = np.random.rand(m, n)
Y = np.random.rand(m, n) * .1
R = np.random.rand(n, n)
gradient = compute_gradient(X, Y, R)
print(f"First row of the gradient matrix: {gradient[0]}")
w4_unittest.test_compute_gradient(compute_gradient)
print('\033[0m', end='')

# UNQ_C5 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003, verbose=True, compute_loss=compute_loss, compute_gradient=compute_gradient):
    '''
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    np.random.seed(129) #makes the random numbers predictable
    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in the word embedding
    R = np.random.rand(X.shape[1], X.shape[1])
    for i in range(train_steps):
        if verbose and i % 25 == 0:
            print(f"loss at iteration {i} is: {compute_loss(X, Y, R):.4f}")
        ### START CODE HERE ###
        # use the function that you defined to compute the gradient
        gradient = compute_gradient(X, Y, R)
        # update R by subtracting the learning rate times gradient
        R -= learning_rate * gradient
        ### END CODE HERE ###
    return R

# Testing your implementation.
np.random.seed(129)
m = 10
n = 5
X = np.random.rand(m, n)
Y = np.random.rand(m, n) * .1
R = align_embeddings(X, Y)
w4_unittest.test_align_embeddings(align_embeddings)
print('\033[0m', end='')

R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)

#k-NN is a method which takes a vector as input and finds the other vectors in the dataset that are closest to it.
#The 'k' is the number of "nearest neighbors" to find (e.g. k=2 finds the closest two neighbors).
#Since we're approximating the translation function from English to French embeddings by a linear transformation matrix 𝐑
#, most of the time we won't get the exact embedding of a French word when we transform embedding 𝐞
#of some particular English word into the French embedding space.
#This is where 𝑘-NN becomes really useful! By using 1-NN with 𝐞𝐑 as input, we can search for an embedding 𝐟 
#(as a row) in the matrix 𝐘 which is the closest to the transformed vector 𝐞𝐑

# UNQ_C8 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def nearest_neighbor(v, candidates, k=1, cosine_similarity=cosine_similarity):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    ### START CODE HERE ###
    similarity_l = []
    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v, row)
        # append the similarity to the list
        similarity_l.append(cos_similarity)
    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l)
    # Reverse the order of the sorted_ids array
    sorted_ids = np.flip(sorted_ids)
    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[:k]
    ### END CODE HERE ###
    return k_idx

# Test your implementation:
v = np.array([1, 0, 1])
candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
print(candidates[nearest_neighbor(v, candidates, 3)])
w4_unittest.test_nearest_neighbor(nearest_neighbor)
print('\033[0m', end='')

# UNQ_C10 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def test_vocabulary(X, Y, R, nearest_neighbor=nearest_neighbor):
    '''
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''
    ### START CODE HERE ###
    # The prediction is X times R
    pred = np.dot(X, R)
    # initialize the number correct to zero
    num_correct = 0
    # loop through each row in pred (each transformed embedding)
    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i], Y)
        # if the index of the nearest neighbor equals the row of i... \
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1
    # accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    accuracy = num_correct / len(X)
    ### END CODE HERE ###
    return accuracy

X_val, Y_val = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)
acc = test_vocabulary(X_val, Y_val, R_train)  # this might take a minute or two
print(f"accuracy on test set is {acc:.3f}")
# Test your function
w4_unittest.unittest_test_vocabulary(test_vocabulary)
print('\033[0m', end='')

#In this part of the assignment, you will implement a more efficient version of k-nearest neighbors using locality sensitive hashing. You will then apply this to document search.

# get the positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

# UNQ_C12 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_document_embedding(tweet, en_embeddings, process_tweet=process_tweet):
    '''
    Input:
        - tweet: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    '''
    doc_embedding = np.zeros(300)
    ### START CODE HERE ###
    # process the document into a list of words (process the tweet)
    processed_doc = process_tweet(tweet)
    for word in processed_doc:
        # add the word embedding to the running total for the document embedding
        if word in en_embeddings:
            doc_embedding = doc_embedding + en_embeddings[word]
    ### END CODE HERE ###
    return doc_embedding

# testing your function
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
tweet_embedding = get_document_embedding(custom_tweet, en_embeddings_subset)
print(tweet_embedding[-5:])
w4_unittest.test_get_document_embedding(get_document_embedding)
print('\033[0m', end='')

# UNQ_C14 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_document_vecs(all_docs, en_embeddings, get_document_embedding=get_document_embedding):
    '''
    Input:
        - all_docs: list of strings - all tweets in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of tweet embeddings.
        - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
    '''
    # the dictionary's key is an index (integer) that identifies a specific tweet
    # the value is the document embedding for that document
    ind2Doc_dict = {}
    # this is list that will store the document vectors
    document_vec_l = []
    for i, doc in enumerate(all_docs):
        ### START CODE HERE ###
        # get the document embedding of the tweet
        doc_embedding = get_document_embedding(doc, en_embeddings)
        # save the document embedding into the ind2Tweet dictionary at index i
        ind2Doc_dict[i] = doc_embedding
        # append the document embedding to the list of document vectors
        document_vec_l.append(doc_embedding)
        ### END CODE HERE ###
    # convert the list of document vectors into a 2D array (each row is a document vector)
    document_vec_matrix = np.vstack(document_vec_l)
    return document_vec_matrix, ind2Doc_dict

document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)
print(f"length of dictionary {len(ind2Tweet)}")
print(f"shape of document_vecs {document_vecs.shape}")
w4_unittest.test_get_document_vecs(get_document_vecs)
print('\033[0m', end='')

#You will now implement locality sensitive hashing (LSH) to identify the most similar tweet.
#Instead of looking at all 10,000 vectors, you can just search a subset to find its nearest neighbors.

N_VECS = len(all_tweets)       # This many vectors.
N_DIMS = len(ind2Tweet[1])     # Vector dimensionality.
print(f"Number of vectors is {N_VECS} and each has {N_DIMS} dimensions.")

#Choosing the number of planes
#* Each plane divides the space to 2 parts.
#* So n planes divide the space into 2^n hash buckets.
#* We want to organize 10,000 document vectors into buckets so that every bucket has about 16 vectors.
#* For that we need 10000/16=625 buckets.
#* We're interested in n, number of planes, so that 2^n= 625. Now, we can calculate n=log2(625) = 9.29 (approx 10).
N_PLANES = 10
# Number of times to repeat the hashing to improve the search.
N_UNIVERSES = 25 #You can think of these as 25 separate ways of dividing up the vector space with a different set of planes.

np.random.seed(0)
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
            for _ in range(N_UNIVERSES)]

# UNQ_C17 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def hash_value_of_vector(v, planes):
    """Create a hash for a vector; hash_id says which random hash to use.
    Input:
        - v:  vector of tweet. It's dimension is (1, N_DIMS)
        - planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
    Output:
        - res: a number which is used as a hash for your vector

    """
    ### START CODE HERE ###
    # for the set of planes,
    # calculate the dot product between the vector and the matrix containing the planes
    # remember that planes has shape (300, 10)
    # The dot product will have the shape (1,10)    
    dot_product = np.dot(v, planes)
    # get the sign of the dot product (1,10) shaped vector
    sign_of_dot_product = np.sign(dot_product)
    # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    # if the sign is 0, i.e. the vector is in the plane, consider the sign to be positive
    h = np.where(sign_of_dot_product < 0, 0, 1)
    # remove extra un-used dimensions (convert this from a 2D to a 1D array)
    h = h.flat
    # initialize the hash value to 0
    hash_value = 0
    n_planes = np.size(planes, 1)
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i        
        hash_value += 2**i * h[i]
    ### END CODE HERE ###
    # cast hash_value as an integer
    hash_value = int(hash_value)
    return hash_value

# Test your function
np.random.seed(0)
idx = 0
planes = planes_l[idx]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
print(f" The hash value for this vector,",
      f"and the set of planes at index {idx},",
      f"is {hash_value_of_vector(vec, planes)}")
w4_unittest.test_hash_value_of_vector(hash_value_of_vector)
print('\033[0m', end='')


# UNQ_C19 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)

# This is the code used to create a hash table: 
# This function is already implemented for you. Feel free to read over it.

### YOU CANNOT EDIT THIS CELL

def make_hash_table(vecs, planes, hash_value_of_vector=hash_value_of_vector):
    """
    Input:
        - vecs: list of vectors to be hashed.
        - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
    Output:
        - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
        - id_table: dictionary - keys are hashes, values are list of vectors id's
                            (it's used to know which tweet corresponds to the hashed vector)
    """
    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    # ALTERNATIVE SOLUTION COMMENT:
    # num_buckets = pow(2, num_of_planes)
    num_buckets = 2**num_of_planes

    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i: [] for i in range(num_buckets)}

    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i: [] for i in range(num_buckets)}

    # for each vector in 'vecs'
    for i, v in enumerate(vecs):
        # calculate the hash value for the vector
        h = hash_value_of_vector(v, planes)

        # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h].append(v) # @REPLACE None

        # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h].append(i) # @REPLACE None

    return hash_table, id_table

# UNQ_C20 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# You do not have to input any code in this cell, but it is relevant to grading, so please do not change anything
planes = planes_l[0]  # get one 'universe' of planes to test the function
tmp_hash_table, tmp_id_table = make_hash_table(document_vecs, planes)

print(f"The hash table at key 0 has {len(tmp_hash_table[0])} document vectors")
print(f"The id table at key 0 has {len(tmp_id_table[0])} document indices")
print(f"The first 5 document indices stored at key 0 of id table are {tmp_id_table[0][0:5]}")

# Creating the hashtables
def create_hash_id_tables(n_universes):
    hash_tables = []
    id_tables = []
    for universe_id in range(n_universes):  # there are 25 hashes
        print('working on hash universe #:', universe_id)
        planes = planes_l[universe_id]
        hash_table, id_table = make_hash_table(document_vecs, planes)
        hash_tables.append(hash_table)
        id_tables.append(id_table)
    
    return hash_tables, id_tables

hash_tables, id_tables = create_hash_id_tables(N_UNIVERSES)

#Implement approximate K nearest neighbors using locality sensitive hashing, to search for documents that are similar to a given document at the index doc_id.
#The approximate_knn function finds a subset of candidate vectors that are in the same "hash bucket" as the input vector 'v'. Then it performs the usual k-nearest neighbors search on this subset (instead of searching through all 10,000 tweets).

# UNQ_C21 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# This is the code used to do the fast nearest neighbor search. Feel free to go over it
def approximate_knn(doc_id, v, planes_l, hash_tables, id_tables, k=1, num_universes_to_use=25, hash_value_of_vector=hash_value_of_vector):
    """Search for k-NN using hashes."""
    #assert num_universes_to_use <= N_UNIVERSES

    # Vectors that will be checked as possible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):
            if doc_id == new_id:
                continue
            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]
                vecs_to_consider_l.append(document_vector_at_i)
                # append the new_id (the index for the document) to the list of ids to consider
                ids_to_consider_l.append(new_id)        
                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)
        ### END CODE HERE ###
    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))
    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)
    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)
    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx] for idx in nearest_neighbor_idx_l]
    return nearest_neighbor_ids

doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]

# Sample
nearest_neighbor_ids = approximate_knn(doc_id, vec_to_search, planes_l, hash_tables, id_tables, k=3, num_universes_to_use=5)
print(f"Nearest neighbors for document {doc_id}")
print(f"Document contents: {doc_to_search}")
print("")
for neighbor_id in nearest_neighbor_ids:
    print(f"Nearest neighbor at document id {neighbor_id}")
    print(f"document contents: {all_tweets[neighbor_id]}")

# Test your function
w4_unittest.test_approximate_knn(approximate_knn, hash_tables, id_tables)
print('\033[0m', end='')
