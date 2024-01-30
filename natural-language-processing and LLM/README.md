# natural-language-processing and LLM
## Table of contents
- [DeepLearning.AI course: Natural Language Processing Specialization](#deeplearningai-course-natural-language-processing-specialization)
  - [Week 1: Sentiment Analysis with logistic regression](#week-1-sentiment-analysis-with-logistic-regression)
    - [Introduction](#Introduction)
    - [Extract features](#Extract-features)
    - [Preprocessing](#Preprocessing)
    - [Create a matrix of features](#Create-a-matrix-of-features)
    - [Overview of logistic regression](#Overview-of-logistic-regression)
  - [Week 2: Sentiment Analysis with Naïve Bayes](#Week-2-Sentiment-Analysis-with-Naïve-Bayes)
    - [Probability](#Probability)
    - [Bayes' Rule](#bayes-rule)
    - [Naïve Bayes](#Naïve-Bayes)
    - [Log Likelihood](#Log-Likelihood)
    - [Training Naïve Bayes](#Training-Naïve-Bayes)
    - [Testing Naïve Bayes](#Testing-Naïve-Bayes)
    - [Applications and assumptions of Naïve Bayes](#Applications-and-assumptions-of-Naïve-Bayes)
    - [Error analysis](#Error-analysis)
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

If doing sentiment analysis of tweets, similarly as before, you need to start by creating a negative and positive frequency dictionary. From this we can extract the total amount of words present in positive and negative tweets. Following this, we will calculate the conditional probability of each word given the class (here positive or negative). For this we will divide the frequency of each word by the amount of words per class. For example if 13 words exist in positive tweets and the word happy comes forward 2 times in positive tweets, the associated conditional probability will equal 2/13. From this we will create a new dictionary of conditional probabilities. Certain words will have a nearly similar conditional probability between classes, those will be of no predictive value.<br>
Laplacian Smoothing is a technique used to avoid a conditional probability of 0 which is problematic when performing the naïve bayes inference condition rule. To calculate a conditional probability we take the frequency of a word in relation to a class and divide it by the total amount of words in a class. With Laplacian Smoothing we add 1 to the frequency of a word relative to a class. And to the total amount of words in a class we add the amount of unique words in a class. Those additions will avoid a probability of 0.

When predicting the sentiment of a tweet we will use the conditional probability dictionary and follow the naïve bayes inference condition rule for binary classification. Each word of the tweet has an associated conditional probability for the positive and negative class. We will take the conditional probability of the positive class divided by the conditional probability of the negative class for each word and multiple them between words. This will give us in the end a number larger than 1 if the prediction says the tweet to be positive and smaller than 1 if the prediction says the tweet to be negative.

#### Log Likelihood
Log likelihoods are logarithms of the previously seen conditional probabilities. They are more convenient to work with, are used in deep learning and NLP.

When taking our conditional probabilities dictionary we can calculate for each word the ratio between its probabilities per class. Thus here we divide the probability of a word being positive by the probability of the word being negative. If the ratio equals 1 we know it has a neutral sentiment while a ratio greater than 1 has a positive sentiment and a ratio smaller than 1 has a negative sentiment.

Underflow consists of numbers being so small a computer cannot handle it. We are at risk of that when multiplying small numbers as we do in Naïve Bayes. To avoid this we use logarithms. Lambda is the logarithm of the probability ratio. A lambda equal to 0 indicates a neutral sentiment, greater than 0 a positive sentiment and smaller than 0 a negative sentiment. From those lambda we can create a lambda dictionary that can be used to calculate the log likelihood by summing the lambdas of each word in a phrase. To get the prediction score we need to sum the log likelihood with log prior which consists of the logarithm of amount of positive tweets divided by amount of negative tweets. If this score equals 0 the sentiment prediction is neutral, higher positive and lower negative.

#### Training Naïve Bayes
1. The first step consists of getting annotated data. Meaning for example tweets and them being annotated as positive or negative. This data will afterwards need to be preprocessed as seen in week 1.
2. Create a frequency dictionary.
3. Transform the frequency dictionary into a conditional probability dictionary.
4. Transform the conditional probability dictionary into a lambda dictionary.
5. Calculate the log prior by taking the logarithm of the amount of positive tweets divided by the amount of negative tweets.

#### Testing Naïve Bayes
We will test by making predictions using lambda dictionary and log prior on test set, after preprocessing that test set of course. We make predictions by summing log prior with log likelihood that we calculated by summing lambda values of associated words. Predictions are then compared with true labels and accuracy score is calculated from that by dividing correct predictions by total amount of made predictions for test set.

When predicting on unseen data it is possible some words are not to be found in lambda dictionary built with training set. Those words will be considered neutral.

#### Applications and assumptions of Naïve Bayes
Naïve Bayes simply predicts using the ratio between conditional probabilities of different classes for each feature. Initially it was often used for information retrievel, based on keywords it is able to filter relevant text from non-relevant text. Next to sentiment analysis it can also be used for author authentification which consists of predicting if a text is written by a certain author. Or it can be used for spam filtering of for example emails. It can also be used for word disambiguity which consists of infering the correct definition of a word, that has multiple, from the context it is being used in. 

Naïve Bayes is fast and simple. Not the most accurate but still robust.

In Naïve Bayes different features are assumed to be independent. Thus for text, different words are considered independent from one another while they are not. Different words in a phrase can be related and them being together would be of predictive value. Also Naïve Bayes assumes training samples to be of similar size between classes which is not always the case. Because of those assumptions we call this method 'naïve'.

#### Error analysis
Removing punctuations during preprocessing can sometimes contribute to errors. For example this ':)' could be of predictive value in sentiment analysis. Also during preprocessing neutral words when combined with other words can be of predictive value and thus should not be removed. For example the neutral word 'not' when combined with 'good' would become 'not good' and indicate a negative sentiment while removing it and leaving 'good' alone would indicate a positive sentiment. Similarly word order can be of importance, if the word 'not' appears before 'good' it would indicate a negative sentiment while if it would appear elsewhere it may be neutral.

Adversarial attacks consist of phrases containing sarcasm, irony, euphemisms... Those can contain words not aligned with the actual sentiment and thus be confusing.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)
* [codecademy - Apply Natural Language Processing with Python](https://www.codecademy.com/learn/paths/natural-language-processing)
