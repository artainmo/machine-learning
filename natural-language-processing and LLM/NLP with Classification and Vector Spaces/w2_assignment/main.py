from utils import process_tweet, lookup
import pdb
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
import nltk
import string
from nltk.tokenize import TweetTokenizer
from os import getcwd
import w2_unittest

nltk.download('twitter_samples')
nltk.download('stopwords')

filePath = f"{getcwd()}/tmp2/"
nltk.data.path.append(filePath)

# get the sets of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# split the data into two pieces, one for training and one for testing (validation set)
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# avoid assumptions about the length of all_positive_tweets
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

#data processing example
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
# print cleaned tweet
print(process_tweet(custom_tweet))

# UNQ_C1 GRADED FUNCTION: count_tweets
def count_tweets(result, tweets, ys):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''
    ### START CODE HERE ###
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)
            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1
            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    ### END CODE HERE ###
    return result

# Testing your function
result = {}
tweets = ['i am happy', 'i am tricked', 'i am sad', 'i am tired', 'i am tired']
ys = [1, 0, 0, 0, 0]
print(count_tweets(result, tweets, ys))
# Test your function
w2_unittest.test_count_tweets(count_tweets)
print('\033[0m', end='')

# Build the freqs dictionary for later uses
freqs = count_tweets({}, train_x, train_y)

# UNQ_C2 GRADED FUNCTION: train_naive_bayes
def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels corresponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0
    ### START CODE HERE ###
    # calculate V, the number of unique words in the vocabulary
    vocab = [key[0] for key in freqs]
    V = len(set(vocab))
    # calculate N_pos, N_neg, V_pos, V_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:
            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += freqs[pair]
        # else, the label is negative
        else:
            # increment the number of negative words by the count for this (word,label) pair
            N_neg += freqs[pair]
    # Calculate D, the number of documents
    D = len(train_y)
    # Calculate D_pos, the number of positive documents
    D_pos = 0
    for y in train_y:
        if y == 1:
            D_pos +=1
    # Calculate D_neg, the number of negative documents
    D_neg = D - D_pos
    # Calculate logprior
    logprior = np.log(D_pos/D_neg)
    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        if (word, 1) in freqs.keys():
            freq_pos = freqs[(word, 1)]
        else:
            freq_pos = 0
        if (word, 0) in freqs.keys():
            freq_neg = freqs[(word, 0)]
        else:
            freq_neg = 0
        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        # calculate the log likelihood of the word
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    ### END CODE HERE ###
    return logprior, loglikelihood

# Use function and test
#logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
#print(logprior)
#print(len(loglikelihood))
# Test your function
#w2_unittest.test_train_naive_bayes(train_naive_bayes, freqs, train_x, train_y)
# Those tests fail on local computer while they don't in jupyter notebook...

# UNQ_C4 GRADED FUNCTION: naive_bayes_predict
def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    ### START CODE HERE ###
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)
    # initialize probability to zero
    p = 0
    # add the logprior
    p += logprior
    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]
    ### END CODE HERE ###
    return p

# test function
my_tweet = 'She smiled.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)
w2_unittest.test_naive_bayes_predict(naive_bayes_predict)
print('\033[0m', end='')
# Experiment with your own tweet.
my_tweet = 'He laughed.'
p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print('The expected output is', p)

# UNQ_C6 GRADED FUNCTION: test_naive_bayes
def test_naive_bayes(test_x, test_y, logprior, loglikelihood, naive_bayes_predict=naive_bayes_predict):
    """
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly
    ### START CODE HERE ###
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0
        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)
    # error is the average of the absolute values of the differences between y_hats and test_y
    error = 0
    for y, pred in zip(test_y, y_hats):
        error += abs(y - pred)
    error /= len(test_y)
    # Accuracy is 1 minus the error
    accuracy = 1 - error  
    ### END CODE HERE ###
    return accuracy

print("Naive Bayes accuracy = %0.4f" % (test_naive_bayes(test_x, test_y, logprior, loglikelihood)))

# test your function
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
    p = naive_bayes_predict(tweet, logprior, loglikelihood)
    # print(f'{tweet} -> {p:.2f} ({p_category})')
    print(f'{tweet} -> {p:.2f}')
w2_unittest.unittest_test_naive_bayes(test_naive_bayes, test_x, test_y)
print('\033[0m', end='')

# UNQ_C8 GRADED FUNCTION: get_ratio
def get_ratio(freqs, word):
    '''
    Input:
        freqs: dictionary containing the words

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    ### START CODE HERE ###
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)
    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)
    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    ### END CODE HERE ###
    return pos_neg_ratio

# Test your function
print(get_ratio(freqs, 'happi'))
w2_unittest.test_get_ratio(get_ratio, freqs)
print('\033[0m', end='')

# UNQ_C9 GRADED FUNCTION: get_words_by_threshold
def get_words_by_threshold(freqs, label, threshold, get_ratio=get_ratio):
    '''
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_list: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}
    ### START CODE HERE ###
    for key in freqs.keys():
        word, _ = key
        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)
        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] >= threshold:
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio
        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio
        # otherwise, do not include this word in the list (do nothing)
    ### END CODE HERE ###
    return word_list

# Test your function: find negative words at or below a threshold
get_words_by_threshold(freqs, label=0, threshold=0.05)
# Test your function; find positive words at or above a threshold
get_words_by_threshold(freqs, label=1, threshold=10)
# Test your function
w2_unittest.test_get_words_by_threshold(get_words_by_threshold, freqs)
print('\033[0m', end='')

# Some error analysis done for you
print('Truth Predicted Tweet')
for x, y in zip(test_x, test_y):
    y_hat = naive_bayes_predict(x, logprior, loglikelihood)
    if y != (np.sign(y_hat) > 0):
        print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
            process_tweet(x)).encode('ascii', 'ignore')))
