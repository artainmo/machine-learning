# In this assignment, you will build an auto-complete system.
# Here are the steps:
#   1. Load and preprocess data
#       * Load and tokenize data.
#       * Split the sentences into train and test sets.
#       * Replace words with a low frequency by an unknown marker <unk>.
#   2. Develop N-gram based language models
#       * Compute the count of n-grams from a given data set.
#       * Estimate the conditional probability of a next word with k-smoothing.
#   3. Evaluate the N-gram models by computing the perplexity score.
#   4. Use your own model to suggest an upcoming word given your sentence.

import math
import random
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')

import w3_unittest
nltk.data.path.append('.')

# Load and display the data
with open("./data/en_US.twitter.txt", "r") as f:
    data = f.read()
print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
display(data[0:300])
print("-------")
print("Last 300 letters of the data")
print("-------")
display(data[-300:])
print("-------")

# Preprocess this data with the following steps:
# 1. Split data into sentences using "\n" as the delimiter.
# 2. Split each sentence into tokens. Note that in this assignment we use "token" and "words" interchangeably.
# 3. Assign sentences into train or test sets.
# 4. Replace tokens that appear less than N times by <unk> and others can be put in vocabulary.

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C1 GRADED_FUNCTION: split_to_sentences ###
def split_to_sentences(data):
    """
    Split data by linebreak "\n"

    Args:
        data: str

    Returns:
        A list of sentences
    """
    ### START CODE HERE ###
    sentences = data.split('\n')
    ### END CODE HERE ###
    # Additional cleaning (This part is already implemented)
    # - Remove leading and trailing spaces from each sentence
    # - Drop sentences if they are empty strings.
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    return sentences

# Test your function
w3_unittest.test_split_to_sentences(split_to_sentences)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C2 GRADED_FUNCTION: tokenize_sentences ###
def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)

    Args:
        sentences: List of strings

    Returns:
        List of lists of tokens
    """
    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []
    ### START CODE HERE ###
    # Go through each sentence
    for sentence in sentences:
        # Convert to lowercase letters
        sentence = sentence.lower()
        # Convert into a list of words
        tokenized = nltk.word_tokenize(sentence)
        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)
    ### END CODE HERE ###
    return tokenized_sentences

# Test your function
w3_unittest.test_tokenize_sentences(tokenize_sentences)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C3 GRADED_FUNCTION: get_tokenized_data ###
def get_tokenized_data(data):
    """
    Make a list of tokenized sentences

    Args:
        data: String

    Returns:
        List of lists of tokens
    """
    ### START CODE HERE ###
    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)
    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        tokenized_sentences.append(nltk.word_tokenize(sentence))
    ### END CODE HERE ###
    return tokenized_sentences

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

print("{} data are split into {} train and {} test set".format(
    len(tokenized_data), len(train_data), len(test_data)))
print("First training sample:")
print(train_data[0])
print("First test sample")
print(test_data[0])

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C4 GRADED_FUNCTION: count_words ###
def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences

    Args:
        tokenized_sentences: List of lists of strings

    Returns:
        dict that maps word (str) to the frequency (int)
    """
    word_counts = {}
    ### START CODE HERE ###
    # Loop through each sentence
    for sentence in tokenized_sentences:
        # Go through each token in the sentence
        for token in sentence:
            # If the token is not in the dictionary yet, set the count to 1
            if token not in word_counts:
                word_counts[token] = 1
            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1
    ### END CODE HERE ###
    return word_counts

# Test your function
w3_unittest.test_count_words(count_words)

# UNIT TEST COMMENT: Candidate for Table Driven Tests 
### UNQ_C5 GRADED_FUNCTION: get_words_with_nplus_frequency ###
def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more
    
    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.
    
    Returns:
        List of words that appear N times or more
    """
    # Initialize an empty list to contain the words that
    # appear at least 'minimum_freq' times.
    closed_vocab = []
    # Get the word couts of the tokenized sentences
    # Use the function that you defined earlier to count the words
    word_counts = count_words(tokenized_sentences) 
    ### START CODE HERE ###
#   UNIT TEST COMMENT: Whole thing can be one-lined with list comprehension
#   filtered_words = None
    # for each word and its count
    for word, cnt in word_counts.items():
        # check that the word's count
        # is at least as great as the minimum count
        if cnt >= count_threshold:
            # append the word to the list
            closed_vocab.append(word)
    ### END CODE HERE ###
    return closed_vocab

# Test your function
w3_unittest.test_get_words_with_nplus_frequency(get_words_with_nplus_frequency)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C6 GRADED_FUNCTION: replace_oov_words_by_unk ###
def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.

    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words

    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """
    # Place vocabulary into a set for faster search
    vocabulary = set(vocabulary)
    # Initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []
    # Go through each sentence
    for sentence in tokenized_sentences:
        # Initialize the list that will contain
        # a single sentence with "unknown_token" replacements
        replaced_sentence = []
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # for each token in the sentence
        for token in sentence:
            # Check if the token is in the closed vocabulary
            if token in vocabulary:
                # If so, append the word to the replaced_sentence
                replaced_sentence.append(token)
            else:
                # otherwise, append the unknown token instead
                replaced_sentence.append(unknown_token)
        ### END CODE HERE ###
        # Append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)
    return replaced_tokenized_sentences

# Test your function
w3_unittest.test_replace_oov_words_by_unk(replace_oov_words_by_unk)

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C7 GRADED_FUNCTION: preprocess_data ###
def preprocess_data(train_data, test_data, count_threshold, unknown_token="<unk>", get_words_with_nplus_frequency=get_words_with_nplus_frequency, replace_oov_words_by_unk=replace_oov_words_by_unk):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.
    Args:
        train_data, test_data: List of lists of strings.
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """
    ### START CODE HERE ###
    # Get the closed vocabulary using the train data
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    # For the train data, replace less common words with "<unk>"
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token)
    # For the test data, replace less common words with "<unk>"
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token)
    ### END CODE HERE ###
    return train_data_replaced, test_data_replaced, vocabulary

# Test your function
w3_unittest.test_preprocess_data(preprocess_data)

minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, 
                                                                        test_data, 
                                                                        minimum_freq)
rint("First preprocessed training sample:")
print(train_data_processed[0])
print()
print("First preprocessed test sample:")
print(test_data_processed[0])
print()
print("First 10 vocabulary:")
print(vocabulary[0:10])
print()
print("Size of vocabulary:", len(vocabulary))

# In this section, you will develop the n-grams language model.

# UNIT TEST COMMENT: Candidate for Table Driven Tests
### UNQ_C8 GRADED FUNCTION: count_n_grams ###
def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    """
    Count all n-grams in the data

    Args:
        data: List of lists of words
        n: number of words in a sequence

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """
    # Initialize dictionary of n-grams and their counts
    n_grams = {}
    ### START CODE HERE ###
    # Go through each sentence in the data
    for sentence in data:
        # prepend start token n times, and  append the end token one time
        sentence = ([start_token] * n) + sentence + [end_token]
        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)
        # Use 'i' to indicate the start of the n-gram
        # from index 0
        # to the last index where the end of the n-gram
        # is within the sentence.
        for i in range(len(sentence) - (n-1)):
            # Get the n-gram from i to i+n
            n_gram = sentence[i:i+n]
            # check if the n-gram is in the dictionary
            if n_gram in n_grams:
                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1
            ### END CODE HERE ###
    return n_grams

# Test your function
w3_unittest.test_count_n_grams(count_n_grams)

### UNQ_C9 GRADED FUNCTION: estimate_probability ###
def estimate_probability(word, previous_n_gram, 
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    
    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter
    
    Returns:
        A probability
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    ### START CODE HERE ###
    # Set the denominator
    # If the previous n-gram exists in the dictionary of n-gram counts,
    # Get its count.  Otherwise set the count to zero
    # Use the dictionary that has counts for n-grams
    if previous_n_gram in n_gram_counts:
        previous_n_gram_count = n_gram_counts[previous_n_gram]
    else:
        previous_n_gram_count = 0
    # Calculate the denominator using the count of the previous n gram
    # and apply k-smoothing
    denominator = previous_n_gram_count + (k * vocabulary_size)
    # Define n plus 1 gram as the previous n-gram plus the current word as a tuple
    n_plus1_gram = previous_n_gram + (word,)
    # Set the count to the count in the dictionary,
    # otherwise 0 if not in the dictionary,
    # use the dictionary that has counts for the n-gram plus current word    
    if n_plus1_gram in n_plus1_gram_counts:
        n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram]
    else:
        n_plus1_gram_count = 0
    # Define the numerator use the count of the n-gram plus current word,
    # and apply smoothing
    numerator = n_plus1_gram_count + k
    # Calculate the probability as the numerator divided by denominator
    probability = numerator / denominator 
    ### END CODE HERE ### 
    return probability

# Test your function
w3_unittest.test_estimate_probability(estimate_probability)

#The function defined below loops over all words in vocabulary to calculate probabilities for all possible words.
def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>",  k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing

    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter

    Returns:
        A dictionary mapping from next words to the probability.
    """
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)
    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + [end_token, unknown_token]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)

        probabilities[word] = probability
    return probabilities

# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
estimate_probabilities(["a"], unigram_counts, bigram_counts, unique_words, k=1)

def make_count_matrix(n_plus1_gram_counts, vocabulary):
    # add <e> <unk> to the vocabulary
    # <s> is omitted since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]

    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))

    # mapping from n-gram to row
    row_index = {n_gram:i for i, n_gram in enumerate(n_grams)}
    # mapping from next word to column
    col_index = {word:j for j, word in enumerate(vocabulary)}

    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count
    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
bigram_counts = count_n_grams(sentences, 2)
print('bigram counts')
display(make_count_matrix(bigram_counts, unique_words))
# The display() function is often encountered in environments like Jupyter Notebooks or IPython, 
# which are popular tools among data scientists, researchers, and educators for interactive computing and data visualization.
# This function isn't part of the standard Python library but is provided by IPython to offer an advanced mechanism for 
# displaying a wide variety of object types beyond what Python's built-in print() function can handle.

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    count_matrix = make_count_matrix(n_plus1_gram_counts, unique_words)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix

sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
bigram_counts = count_n_grams(sentences, 2)
print("bigram probabilities")
display(make_probability_matrix(bigram_counts, unique_words, k=1))

# In this section, you will generate the perplexity score to evaluate your model on the test set.

# UNQ_C10 GRADED FUNCTION: calculate_perplexity
def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, start_token='<s>', end_token = '<e>', k=1.0):
    """
    Calculate perplexity for a list of sentences
    
    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant
    
    Returns:
        Perplexity score
    """
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])
    # prepend <s> and append <e>
    sentence = [start_token] * n + sentence + [end_token]
    # Cast the sentence from a list to a tuple
    sentence = tuple(sentence)
    # length of sentence (after adding <s> and <e> tokens)
    N = len(sentence)
    # The variable p will hold the product
    # that is calculated inside the n-root
    # Update this in the code below
    product_pi = 1.0
    ### START CODE HERE ###
    # Index t ranges from n to N - 1, inclusive on both ends
    for t in range(n, N):
        # get the n-gram preceding the word at position t
        print(n)
        n_gram = sentence[t-n:t]
        # get the word at position t
        word = sentence[t]
        # Estimate the probability of the word given the n-gram
        # using the n-gram counts, n-plus1-gram counts,
        # vocabulary size, and smoothing constant
        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)
        # Update the product of the probabilities
        # This 'product_pi' is a cumulative product 
        # of the (1/P) factors that are calculated in the loop
        product_pi *= 1/probability
        ### END CODE HERE ###
    # Take the Nth root of the product
    perplexity = (product_pi)**(1/N)
    ### END CODE HERE ### 
    return perplexity

# test your code
sentences = [['i', 'like', 'a', 'cat'],
                 ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))
unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
perplexity_train = calculate_perplexity(sentences[0],
                                         unigram_counts, bigram_counts,
                                         len(unique_words), k=1.0)
print(f"Perplexity for first train sample: {perplexity_train:.4f}")
test_sentence = ['i', 'like', 'a', 'dog']
perplexity_test = calculate_perplexity(test_sentence,
                                       unigram_counts, bigram_counts,
                                       len(unique_words), k=1.0)
print(f"Perplexity for test sample: {perplexity_test:.4f}")

# Test your function
w3_unittest.test_calculate_perplexity(calculate_perplexity)

# In this section, you will combine the language models developed so far to implement an auto-complete system.

# Compute probabilities for all possible next words and suggest the most likely one.

# UNQ_C11 GRADED FUNCTION: suggest_a_word
def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, end_token='<e>', unknown_token="<unk>", k=1.0, start_with=None):
    """
    Get suggestion for the next word

    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length >= n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A tuple of
          - string of the most likely next word
          - corresponding probability
    """
    # length of previous words
    n = len(list(n_gram_counts.keys())[0])
    # append "start token" on "previous_tokens"
    previous_tokens = ['<s>'] * n + previous_tokens
    # From the words that the user already typed
    # get the most recent 'n' words as the previous n-gram
    previous_n_gram = previous_tokens[-n:]
    # Estimate the probabilities that each word in the vocabulary
    # is the next word,
    # given the previous n-gram, the dictionary of n-gram counts,
    # the dictionary of n plus 1 gram counts, and the smoothing constant
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)
    # Initialize suggested word to None
    # This will be set to the word with highest probability
    suggestion = None
    # Initialize the highest word probability to 0
    # this will be set to the highest probability
    # of all words to be suggested
    max_prob = 0
    ### START CODE HERE ###
    # For each word and its probability in the probabilities dictionary:
    for word, prob in probabilities.items():
        # If the optional start_with string is set
        if start_with:
            # Check if the beginning of word does not match with the letters in 'start_with'
            if not word.startswith(start_with):
                # if they don't match, skip this word (move onto the next word)
                continue
        # Check if this word's probability
        # is greater than the current maximum probability
        if prob > max_prob:
            # If so, save this word as the best suggestion (so far)
            suggestion = word
            # Save the new maximum probability
            max_prob = prob
    ### END CODE HERE
    return suggestion, max_prob

# The function defined below loops over various n-gram models to get multiple suggestions.

def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions

# test your code
sentences = [['i', 'like', 'a', 'cat'],
             ['this', 'dog', 'is', 'like', 'a', 'cat']]
unique_words = list(set(sentences[0] + sentences[1]))

unigram_counts = count_n_grams(sentences, 1)
bigram_counts = count_n_grams(sentences, 2)
trigram_counts = count_n_grams(sentences, 3)
quadgram_counts = count_n_grams(sentences, 4)
qintgram_counts = count_n_grams(sentences, 5)

n_gram_counts_list = [unigram_counts, bigram_counts, trigram_counts, quadgram_counts, qintgram_counts]
previous_tokens = ["i", "like"]
tmp_suggest3 = get_suggestions(previous_tokens, n_gram_counts_list, unique_words, k=1.0)

print(f"The previous words are 'i like', the suggestions are:")
display(tmp_suggest3)
# Output: [('a', 0.2727272727272727), ('a', 0.2), ('a', 0.2), ('a', 0.2)]

# Let's see this with n-grams of varying lengths (unigrams, bigrams, trigrams, 4-grams...6-grams).

n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)

previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
display(tmp_suggest4)
# Output: [('be', 0.027665685098338604),
#           ('have', 0.00013487086115044844),
#           ('have', 0.00013490725126475548),
#           ('i', 6.746272684341901e-05)]
