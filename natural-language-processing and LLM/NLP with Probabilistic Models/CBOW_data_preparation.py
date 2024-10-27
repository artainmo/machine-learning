# In the data preparation phase, starting with a corpus of text, you will:
# * Clean and tokenize the corpus.
# * Extract the pairs of context words and center word that will make up the training data set for the CBOW model. The context words are the features that will be fed into the model, and the center words are the target values that the model will learn to predict.
# * Create simple vector representations of the context words (features) and center words (targets) that can be used by the neural network of the CBOW model.

import re #For regex operations
import nltk #natural language tool kit

nltk.download('punkt') 

import emoji
import numpy as np
from nltk.tokenize import word_tokenize
from utils2 import get_dict

# Define a corpus
corpus = 'Who ❤️ "word embeddings" in 2020? I do!!!'

# replace all interrupting punctuation signs — such as commas and exclamation marks — with periods.
data = re.sub(r'[,!?;-]+', '.', corpus)
# Print cleaned corpus
print(f'After cleaning punctuation:  {data}')
# Output: Who ❤️ "word embeddings" in 2020. I do.

# Tokenize the cleaned corpus
data = nltk.word_tokenize(data)
# Print the tokenized version of the corpus
print(f'After tokenization:  {data}')
# Output: ['Who', '❤️', '``', 'word', 'embeddings', "''", 'in', '2020', '.', 'I', 'do', '.']

# Define the 'tokenize' function that will include the steps previously seen
def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)  # tokenize string to words
    data = [ ch.lower() for ch in data
             if ch.isalpha()
             or ch == '.'
             or emoji.get_emoji_regexp().search(ch)
           ]
    return data

# Define new corpus
corpus = 'I am happy because I am learning'
# Save tokenized version of corpus into 'words' variable
words = tokenize(corpus)
# Print the tokenized version of the corpus
print(f'Words (tokens):  {words}')
# Output: ['i', 'am', 'happy', 'because', 'i', 'am', 'learning']

def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[(i - C):i] + words[(i+1):(i+C+1)]
        yield context_words, center_word
        i += 1

# Print 'context_words' and 'center_word' for the new corpus with a 'context half-size' of 2
for x, y in get_windows(['i', 'am', 'happy', 'because', 'i', 'am', 'learning'], 2):
    print(f'{x}\t{y}')
# Output:
# ['i', 'am', 'because', 'i']	happy
# ['am', 'happy', 'i', 'am']	because
# ['happy', 'because', 'am', 'learning']	i

# To finish preparing the training set, you need to transform the context words and center words into vectors.

# To create one-hot word vectors, you can start by mapping each unique word to a unique integer (or index). We have provided a helper function, get_dict, that creates a Python dictionary that maps words to integers and back.
# Get 'word2Ind' and 'Ind2word' dictionaries for the tokenized corpus
word2Ind, Ind2word = get_dict(words)
print(word2Ind)
# Output: {'am': 0, 'because': 1, 'happy': 2, 'i': 3, 'learning': 4}
print(Ind2word)
# Output: {0: 'am', 1: 'because', 2: 'happy', 3: 'i', 4: 'learning'}

# Save index of word 'happy' into the 'n' variable
n = word2Ind['happy']
# Create vector with the same length as the vocabulary, filled with zeros
center_word_vector = np.zeros(V)
# Replace element number 'n' with a 1
center_word_vector[n] = 1

# Define the 'word_to_one_hot_vector' function that will include the steps previously seen
def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector

word_to_one_hot_vector('happy', word2Ind, V)

#To create the vectors that represent context words, you will calculate the average of the one-hot vectors representing the individual words.

# Define list containing context words
context_words = ['i', 'am', 'because', 'i']
# Create one-hot vectors for each context word using list comprehension
context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
# Compute mean of the vectors using numpy
np.mean(context_words_vectors, axis=0) #Note the axis=0 parameter that tells mean to calculate the average of the rows

# Define the 'context_words_to_vector' function that will include the steps previously seen
def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors

context_words_to_vector(['i', 'am', 'because', 'i'], word2Ind, V)

# Define the generator function 'get_training_example'
def get_training_example(words, C, word2Ind, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2Ind, V), word_to_one_hot_vector(center_word, word2Ind, V)

# Print vectors associated to center and context words for corpus using the generator function
for context_words_vector, center_word_vector in get_training_example(words, 2, word2Ind, V):
    print(f'Context words vector:  {context_words_vector}')
    print(f'Center word vector:  {center_word_vector}')
    print()
# Output:
# Context words vector:  [0.25 0.25 0.   0.5  0.  ]
# Center word vector:  [0. 0. 1. 0. 0.]
# Context words vector:  [0.5  0.   0.25 0.25 0.  ]
# Center word vector:  [0. 1. 0. 0. 0.]
# Context words vector:  [0.25 0.25 0.25 0.   0.25]
# Center word vector:  [0. 0. 0. 1. 0.]

# Your training set is ready, you can now move on to the CBOW model itself which will be covered in the next lecture notebook.
