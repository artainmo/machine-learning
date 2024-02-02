#In this assignment we will explore word vectors. In natural language processing, we represent each word as a vector consisting of numbers. The vector encodes the meaning of the word. 
#Rather than make you code the machine learning models from scratch, we will show you how to use them. In the real world, you can always load the trained word vectors, and you will almost never have to train them from scratch.

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import w3_unittest
from utils import get_vectors

data = pd.read_csv('./data/capitals.txt', delimiter=' ')
data.columns = ['city1', 'country1', 'city2', 'country2']
# print first five elements in the DataFrame
print(data.head(5))

# Note that because the original google news word embedding dataset is about 3.64 gigabytes, the workspace is not able to handle the full file set. So we've downloaded the full dataset, extracted a sample of the words that we're going to analyze in this assignment, and saved it in a pickle file called word_embeddings_capitals.p
word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb"))
print(len(word_embeddings))  # there should be 243 words that will be used in this assignment
print("dimension: {}".format(word_embeddings['Spain'].shape[0]))

# UNQ_C1 GRADED FUNCTION: cosine_similarity
def cosine_similarity(A, B):
    '''
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''
    ### START CODE HERE ###
    dot = np.dot(A, B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = dot / (norma * normb)
    ### END CODE HERE ###
    return cos

# feel free to try different words
king = word_embeddings['king']
queen = word_embeddings['queen']
print(cosine_similarity(king, queen))
# Test your function
w3_unittest.test_cosine_similarity(cosine_similarity)
print('\033[0m', end='')

# UNQ_C2 GRADED FUNCTION: euclidean
def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """
    ### START CODE HERE ###
    d = np.linalg.norm(A-B)
    ### END CODE HERE ###
    return d

# Test your function
print(euclidean(king, queen))
w3_unittest.test_euclidean(euclidean)
print('\033[0m', end='')

# UNQ_C3 GRADED FUNCTION: get_country
def get_country(city1, country1, city2, embeddings, cosine_similarity=cosine_similarity):
    """
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their emmbeddings
    Output:
        country: a tuple with the most likely country and its similarity score
    """
    ### START CODE HERE ###
    # store the city1, country 1, and city 2 in a set called group
    group = {city1, country1, city2}
    # get embeddings of city 1, country 1 and city 2
    city1_emb = embeddings[city1]
    country1_emb = embeddings[country1]
    city2_emb = embeddings[city2]
    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # Remember: King - Man + Woman = Queen
    vec = (country1_emb - city1_emb) + city2_emb
    # Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    similarity = -1
    # initialize country to an empty string
    country = ''
    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():
        # first check that the word is not already in the 'group'
        if word not in group:
            # get the word embedding
            word_emb = embeddings[word]
            # calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            cur_similarity = cosine_similarity(vec, word_emb)
            # if the cosine similarity is more similar than the previously best similarity...
            if cur_similarity > similarity:
                # update the similarity to the new, better similarity
                similarity = cur_similarity
                # store the country as a tuple, which contains the word and the similarity
                country = (word, similarity)
    ### END CODE HERE ###
    return country

# Testing your function
print(get_country('Athens', 'Greece', 'Cairo', word_embeddings))
w3_unittest.test_get_country(get_country)
print('\033[0m', end='')

# UNQ_C4 GRADED FUNCTION: get_accuracy
def get_accuracy(word_embeddings, data, get_country=get_country):
    '''
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas DataFrame containing all the country and capital city pairs

    '''
    ### START CODE HERE ###
    # initialize num correct to zero
    num_correct = 0
    # loop through the rows of the dataframe
    for i, row in data.iterrows():
        # get city1
        city1 = data['city1'][i]
        # get country1
        country1 = data['country1'][i]
        # get city2
        city2 = data['city2'][i]
        # get country2
        country2 = data['country2'][i]
        # use get_country to find the predicted country2
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)
        # if the predicted country2 is the same as the actual country2...
        if predicted_country2 == country2:
            # increment the number of correct by 1
            num_correct += 1
    # get the number of rows in the data dataframe (length of dataframe)
    m = len(data)
    # calculate the accuracy by dividing the number correct by m
    accuracy = num_correct / m
    ### END CODE HERE ###
    return accuracy

# Test your function
accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")
w3_unittest.test_get_accuracy(get_accuracy, data)
print('\033[0m', end='')

# UNQ_C5 GRADED FUNCTION: compute_pca


def compute_pca(X, n_components=2):
    """
    Input:
        X: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output:
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    ### START CODE HERE ###
    # mean center the data
    X_demeaned = X - np.mean(X)
    # calculate the covariance matrix
    covariance_matrix = np.cov(X_demeaned, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)
    # sort eigenvalue in increasing order (get the indices from the sort)
    idx_sorted = np.argsort(eigen_vals)
    # reverse the order so that it's from highest to lowest.
    idx_sorted_decreasing = np.flip(idx_sorted)
    # sort the eigen values by idx_sorted_decreasing
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]
    # sort eigenvectors using the idx_sorted_decreasing indices
    eigen_vecs_sorted = eigen_vecs[:, idx_sorted_decreasing]
    # select the first n eigenvectors (n is desired dimension of rescaled data array, or n_components)
    eigen_vecs_subset = eigen_vecs_sorted[:, :n_components]
    # transform the data by multiplying the transpose of the eigenvectors with the transpose of the de-meaned data
    # Then take the transpose of that product.
    X_reduced = np.dot(X_demeaned, eigen_vecs_subset)
    ### END CODE HERE ###
    return X_reduced

# Testing your function
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)
w3_unittest.test_compute_pca(compute_pca)
print('\033[0m', end='')

#plotting
words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']
X = get_vectors(word_embeddings, words)
print('You have 11 words each of 300 dimensions thus X.shape is:', X.shape)
result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))
plt.show() 
#The word vectors for gas, oil and petroleum appear related to each other, because their vectors are close to each other. Similarly, sad, joyful and happy all express emotions, and are also near each other.
#Similar words tend to be clustered near each other. Sometimes, even antonyms tend to be clustered near each other. Antonyms describe the same thing but just tend to be on the other end of the scale. They are usually found in the same location of a sentence, have the same parts of speech, and thus when learning the word vectors, you end up getting similar weights.
