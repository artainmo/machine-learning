# natural-language-processing and LLM
## Table of contents
- [DeepLearning.AI course: Natural Language Processing Specialization](#deeplearningai-course-natural-language-processing-specialization)
- [Resources](#Resources)

## DeepLearning.AI course: Natural Language Processing Specialization
### Week 1: Sentiment Analysis with logistic regression
#### Introduction
In the first week we will learn about logistic regression. Logistic regression takes as inputs features (X) and labels (Y). It uses parameters (theta) to make predictions (Yhat). While training, parameters are updated in relation to the error (cost) in Yhat compared to Y, until the cost becomes minimized.

In example exercise we will take a tweet and predict if it has a positive or negative sentiment. Positive sentiment tweets will have a label Y of 1 and negative sentiment tweets a label Y of 0. First we will need to extract features, train model and lastly classify the tweets by making predictions with trained model.

#### Extract features
To extract features, text will have to get encoded as an array of numbers. For that, first you need to create a vocabulary list that consists of all the unique words in the read/learned text(s). This vocabulary list will represent a list of all the features (X). For the example text to learn/predict on replace each word in vocabulary list that is present in that text with 1 else 0. This is how we give the correct inputs to the features. However as the vocabulary list gets long, the features list too making the algorithm slower.<br>
You want to use as feature the amount of times a word comes forward in example text. For this we will create a frequency dictionary, in this case one for positive and negative sentiment. For all the training examples with negative sentiment count the number of times each vocabulary list word comes forward inside the negative frequency dictionary and for all training examples with positive sentiment count the number of times each vocabulary list word comes forward inside the positive frequency dictionary. Now those frequency dictionaries can be used to extract useful features, limiting the size of X to 3. The first value of X will be the bias value of 1, second we will use the sum of all positive frequencies and lastly sum of all negative frequencies. This means that if our vocabulary list for example contains the word sad, sad could be equal to 3 in our negative frequency dictionary but equal 0 in our positive frequency dictionary. If the text being evaluated also contains the word sad it will sum 0 in positive frequency dictionary but sum 3 in negative frequency dictionary, thus third feature consisting of negative frequency dictionary sum will probably be of higher value.

#### Preprocessing
You will learn how to use 'stemming' and 'stop words' to preprocess texts.

'Stop words' preprocessing consists of removing words with no significant meaning. 'Stop words' lists are given and contain words such as 'and', 'is', 'are', 'at'... Punctuations can also be removed but sometimes is best kept as it can also be of predictive value. Tweets often contain handles and URLs who have no predictive power and thus can be removed too.

Stemming in NLP consists of transforming each word into its base stem, which means the set of characters that are used to construct a word and its derivatives. For example the stem of word 'tuning' is 'tun' because its derivates contain 'tune' but also 'tuning'. All words can also be set to lowercase.

#### Create a matrix of features
Each tweet in above described example can be described as a vector of 3 after feature extraction using frequency dictionaries. Multiple tweets can be described as rows of vectors with size 3 in a matrix.

The general implementation would consist of building the frequency dictionary, initializing the matrix X with number of tweets as size of rows and 3 as column size. Subsequently the tweets should be preprocessed and features extracted to fill the matrix X's rows.

#### Overview of logistic regression
The features X uses the sigmoid function together with parameters theta to make predictions. Sigmoid outputs a value between 0 and 1. If prediction is value closer to 0 than 1 then it equals negative sentiment and vice-versa.

While training we need to find the theta values that minimizes the cost. To do that theta needs to be updated in the direction of the gradient of the cost function with the goal of going towards cost function minima. Thus first we need to predict with sigmoid, afterwards get gradient from cost, and substract that gradient from theta. This should be done a number of times until theta converges to cost function local minima.

Data should be split in training and test data set. The training set is used for training and the test set is used after training to see if predictions generalize well on previously unseen data.

### Week 2: Sentiment Analysis with Naïve Bayes
#### Probability
Probability is of fundamental value in NLP. One way of evaluating probabilities is by counting how frequently an event occurs among other events. For example the probability of a tweet being positive (P(positive)) is equal to the amount of positive tweets divided by the total amount of tweets. 

Conditional probabilities add a condition, for example what is the probability that a positive tweet contains the word 'happy' (P("happy"|positive)). The answer is the total amount of positive and 'happy' containing tweets divided by the total amount of positive tweets. Conditional probabilities basically evaluate the probability of event B given that A happened (P(A|B)) and are calculated by dividing A and B containing events (P(A ∩ B)) by B containing events (P(B)).

#### Bayes' Rule
Bayes' rule is derived from the above described conditional probabilities' formula and goes as follows, the probability of X given Y (P(X|Y)) is equal to the probability of Y given X (P(Y|X)) times the ratio of X (P(X)) over Y (P(Y)).

The complete formula:
```
P(X|Y) = P(Y|X) x P(X)/P(Y)
```

The Bayes' rule will be used at multiple occasions in NLP. It allows to find the probability of X given Y (P(X|Y)) if the probability of Y given X (P(Y|X)) is already known.

#### Naïve Bayes
Naïve bayes is a quick and simple classifier that can be used in NLP. It is a supervised learning method that shares similarities with previously reviewed logistic regression. It is called naïve because it makes the assumption that all features are independent which in reality is rarely the case.

If doing sentiment analysis of tweets, similarly as before, you need to start by creating a negative and positive frequency dictionary. From this we can extract the total amount of words present in positive and negative tweets. Following this, we will calculate the conditional probability of each word given the class (here positive or negative). For this we will divide the frequency of each word by the amount of words per class. For example if 13 words exist in positive tweets and the word happy comes forward 2 times in positive tweets, the associated conditional probability will equal 2/13. From this we will create a new dictionary of conditional probabilities. Certain words will have a nearly similar conditional probability between classes, those will be of no predictive value.

When predicting the sentiment of a tweet we will use the conditional probability dictionary and follow the naïve bayes inference condition rule for binary classification. Each word of the tweet has an associated conditional probability for the positive and negative class. We will take the conditional probability of the positive class divided by the conditional probability of the negative class for each word and multiple them between words. This will give us in the end a number larger than 1 if the prediction says the tweet to be positive and smaller than 1 if the prediction says the tweet to be negative.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)
* [codecademy - Apply Natural Language Processing with Python](https://www.codecademy.com/learn/paths/natural-language-processing)
