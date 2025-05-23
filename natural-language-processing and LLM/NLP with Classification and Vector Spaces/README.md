# Natural Language Processing with Classification and Vector Spaces
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Classification and Vector Spaces](#deeplearningai-natural-language-processing-specialization-nlp-with-classification-and-vector-spaces)
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
  - [Week 3: Vector Space Models](#Week-3-Vector-Space-Models)
    - [Vector space models](#Vector-space-models)
    - [Construct-vectors](#Construct-vectors)
    - [Euclidian distance and cosine similarity](#Euclidian-distance-and-cosine-similarity)
    - [Manipulating word vectors](#Manipulating-word-vectors)
    - [Visualization and PCA](#Visualization-and-PCA)
  - [Week 4: Machine Translation and Document Search](#Week-4-Machine-Translation-and-Document-Search)
    - [Introduction](#introduction-1)
    - [Transforming word vectors](#Transforming-word-vectors)
    - [K-nearest neighbors and hashing](#k-nearest-neighbors-and-hashing)
    - [Searching documents](#Searching-documents)
- [Resources](#Resources)

## DeepLearning.AI: Natural Language Processing Specialization: NLP with Classification and Vector Spaces
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

While training we need to find the theta values that minimizes the cost. To do that theta needs to be updated in the direction of the gradient of the cost function with the goal of going towards cost function minima. Thus first we need to predict with sigmoid, afterwards get gradient from cost, and substract that gradient from theta. This should be done a number of times until theta converges to cost function minima.

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

When predicting the sentiment of a tweet we will use the conditional probability dictionary and follow the naïve bayes inference condition rule for binary classification. Each word of the tweet has an associated conditional probability for the positive and negative class. We will take the conditional probability of the positive class divided by the conditional probability of the negative class for each word and multiply them across all words in the tweet. This will give us in the end a number larger than 1 if the prediction says the tweet to be positive and smaller than 1 if the prediction says the tweet to be negative.

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

### Week 3: Vector Space Models
#### Vector space models
Representing vector spaces as word vectors is fundamental to NLP because all NLP models start by transforming text/words into numerical encodings.

Vector space models help identify if word groups share similar meanings even if they don't consist of similar words. For example 'What is your age?' and 'How old are you?' share similar meanings but not composition of words.<br>
Vector space models allow capturing dependencies between words. For example in the phrase 'You eat cereal from a bowl', cereal is dependent to bowl, or in the phrase 'you buy something and someone else sells it' the action of selling depends on the initial action of buying.

Vector space models are used in information extraction to answer basic questions, but also in machine translation (AIs translating texts from one human language to another) and chatbots among other applications.

Vector space models are able to identify the context around each word in a text, which allows the identification of the relative meaning of the text.

#### Construct vectors
Words or documents can be encoded as vectors who are constructed based off a co-occurrence matrix. A co-occurrence matrix can capture relationships and patterns between items within a given dataset. 

The co-occurrence of two different words is the number of times they appear in the corpus together within a certain word distance (k). If words are right next to each other, k is equal to 1, if one word sits between, k is equal to 2...<br>
For example if we have the two folowing tweets: '*I like simple data*' and '*I prefer simple raw data*'. A word-by-word co-occurence matrix with k equal to 2 would look like:

|      | simple | raw | like | prefer | I   |
| ---- | ------ | --- | ---- | ------ | --- |
| data | 2      | 1   | 1    | 0      | 0   |

In natural language processing a co-occurrence matrix often represents the occurrence of words within a specific context. Each cell in the matrix represents the number of times two items (such as words) appear together within a predefined window of text.

For a word-by-document design you need to count the occurence of a word in documents that belong to a certain category. For example you could have a corpus consisting of documents belonging to categories such as entertainment, economy or machine-learning. To create a word-by-document occurence matrix you would have to indicate the number of times each word appears in each category of documents.

|      | entertainment | economy | machine-learning |
| ---- | ------------- | ------- | ---------------- |
| data | 500           | 6620    | 9320             |
| film | 7000          | 4000    | 1000             |

By comparing the values between columns you can extract the degree of similarity or other relationships between those columns.

#### Euclidian distance and cosine similarity
Euclidian distance is a similarity metric that allows to identify how far two points or two vectors are from each other.

Vectors can be represented as points in the vector space. The distance between those points gives us the Euclidian distance. The distance between two points is calculated with the following formula `d=√((x2 – x1)² + (y2 – y1)²)`. To get the Euclidian distance of n-dimensional vectors we need to loop over all the dimensions and sum their distance as indicated with following formula `√(N∑i=1(vi-wi)²)`. If we take the above table and want to calculate the Euclidian distance between the economy and machine-learning columns, we would get √((9320-6620)² + (1000-4000)²). This formula is same as the norm of the difference between vectors which can be implemented in python as follows `d = np.linalg.norm(v-w)`. 

Cosine similarity is another similarity metric and uses the cosine of the angle between two vectors which indicates if those vectors are close to each other or not, thus how similar they are.

Euclidean distance as a similarity metric can lack precision when documents are of different sizes. Cosine similarity can better represent the similarity between two vectors representing documents of different sizes because it uses angles and thus is not dependent on the size of the corpus.

To calculate the cosine similarity we use the dot product and norm of vectors. The norm or magnitude of a vector is calculated with the square root of the sum of its squared elements `||v|| = √(N∑i=1(vi²))`. The dot product is the sum of elements within the product of two vectors `v.w = N∑i=1(vi.wi)`. The cosine of the angle between two vectors is calculated by taking the dot product between those vectors and dividing it by the product of the norm of those two vectors `cos(β) = (v.w) / (||v|| ||w||)`. A cosine of value 0 equates an angle of 90°, a cosine of value 1 equates an angle of 0°. Thus the closer to 1 a cosine, the more similar the two vectors.

#### Manipulating word vectors
By the use of known relationships between words we can find unknown relationships between other words. This is possible due to vector spaces capturing the relative meaning of words within a context.

If one word vector represents the USA, another its capital Washington and we know the word vector of Russia but not its capital, then we can find the capital of Russia via the relationship between the USA and Washington. By substracting the word vector Washington with word vector USA we get a vector representing their relationship. If we addition this relationship vector with word vector Russia we should get the word vector representing Russia's capital. However if no existing word vector is to be found there, we need to find the closest one to that vector using cosine similarity or Euclidian distance. Then we should find Moscow as being the closest word vector and thus the capital of Russia.

#### Visualization and PCA
Often, word vectors will contain a high amount of dimensions. You want to reduce those dimensions to 2 if you want to visualize them by plotting them on an XY axis. Principal component analysis (PCA) is an unsupervised dimensionality reduction algorithm allowing this.

Visualization can help see the relationship between words in the vector space. When plotting, note that words with similar part of speech (POS) tags are next to one another. POS tags indicate a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context. Words with similar POS come together because many of the training algorithms learn words by identifying the neighboring words.

Covariance is the mean value of the product of the deviations of two variates from their respective means. A positive covariance indicates that both variates tend to be high or low at same time (similar direction) while a negative covariance indicates that if one variate is high the other will be low (opposite direction).
A covariance matrix is a square matrix giving the covariance between each element pair in a vector.

PCA uses the Eigenvalues and Eigenvectors of our covariance matrix to reduce dimensions while retaining as much salient information as possible. Eigenvectors of covariance matrices give directions of uncorrelated features and Eigenvalues indicate the variance of your data in each of these new features.

For PCA the first step is to get a set of uncorrelated features. You may normalize your data. Then get your covariance matrix and perform a singular value decomposition (SVD) to get a set of three matrices. The first matrix will contain the Eigenvectors stacked column-wise, and the second matrix holds the Eigenvalues on its diagonal. SVD is already implemented in many programming libraries. The next step is to project our data on a new set of features. First you will perform the dot product between the normalized data and the first n (n=2 if wanting to plot on XY axis) columns of the Eigenvectors matrix. The Eigenvector is used as a transformation matrix that will maximize the variance of one class and minimize the variance of another class with the goal of maximizing the differences between the two classes. Eigenvectors and Eigenvalues should be organized by the Eigenvalues in descending order. This will allow us to select the first n eigenvectors. Because they come first they will be associated with the highest eigenvalue magnitudes and thus be the most discriminative power. However most libraries order those matrices for you.

### Week 4: Machine Translation and Document Search
#### Introduction
We will learn how to transform word vectors, implement k-nearest neighbors which is an algorithm searching for similar items, hash tables which helps assign word vectors into subsets, divide the vector space into regions to help search of similar words, finally we will implement locality sensitive hashing which helps perform the approximated k-nearest neighbors to search for similar word vectors. 

Knowing how to find similar word vectors will allow the implementation of machine translation and document search. Machine translation consists of translating text from one language to another while document search consists of finding phrases/documents with similar meanings. 

#### Transforming word vectors
If wanting to translate an English word to French, we need to transform the English word vector into the associated French word vector. Next you need to search the most similar word vector in the French word vector space to the transformed word vector.

The transformation can be done with a matrix we will call R. To get R we will need to train a supervised learning algorithm. Thus we will need a subset of English words with their French translations to be able to find R with R allowing us to find/predict all the other translations afterwards.<br>
If X is a matrix containing the English word vectors and Y a matrix containing the associated French word vectors, then `XR ≈ Y`. To find R we can use the loss function `Loss = ||XR - Y||` (The `|| ||` symbols refers to the norm or magnitude) and start with an R of random values. As the loss diminishes the R matrix improves. To improve R, take the gradient of the loss function `g = (2/m)(XT (XR - Y))` (with m equalling the number of training examples, and XT being the transpose of X) and substract it to R with a step size α `R = R - αg`. You can loop to improve R until the loss is small enough.

#### K-nearest neighbors and hashing
A transformed English word vector ends in the French word vector space. But won't necessarily precisely equate a word in that French word vector space. Thus we need to find from the French word vectors the one most similar to the transformed vector. The k-nearest neighbor of the transformed vector will help us find that matching word by looking at the k most similar ones. Here we will use a value of 1 for k, which means we will search the most similar one. To evaluate similarity we can use cosine similarity reviewed previously.

Instead of iterating over all the words, it can be beneficial to only search the words in a specific region we know contains the word we are looking for. Hashing helps us only look at words who are of a certain category. Hashing thus improves lookup speed.

A hash function assigns to a vector a hash value. This hash value is used to categorize the vector. One hash function is to take the vector, divide it by the number of categories and take the remainder of that division which will equate the hash value. The hash value will indicate the category number. For example if the remainder and thus hash value is 0, then the vector will be assigned to the first category 0. This hash function is not ideal however for our current demand. Instead we would want a hash function that categorizes based on location/proximity. We call this locality sensitive hashing. 

A hash table is the dictionary containing the categories as keys and assigned vectors as values.

Locality sensitive hashing will divide the vector space using lines. The dot product between a word vector on the vector space and one of those dividing lines will give a number that when positive indicates the point is on one side of the line, if negative on the opposite side of the line and if 0 is on the line. This can help locate word vectors. Now as we go to dimensions higher than 2, we would be using planes instead of lines. To calculate a hash value you will want to calculate on what side of each plane a word vector lies. If on positive side or on plane we will assign value 1, else 0. Thus when the dot product between plane and word vector equals 0 or more we assign value 1, else value 0. This assigned value for each plane we call h. Finally we will use those h values to calculate the hash value like this `hash_value = 2^0*h0 + 2^1*h1 + 2^2*h2 + 2^3*h3...` thus the formula can be written like this `hash_value = H∑i(2^i * hi)`.

With 'approximate' nearest neighbors you only search within a subset of all the vectors. You trade some precision for speed of search. Locality sensitive hashing is used to find the neighbors of a vector.<br> 
When creating planes to divide the vector space we cannot know for sure what planes would be best. This is why we use multiple sets of randomly generated planes. Each set of planes creates own categories. If for certain plane sets two vectors are in different categories for another plane set they may be in same category. Taking all the neighbouring vectors found in the different categories from the different plane sets we get a more complete set of neighbours.<br>
You can generate one set of random planes like this `random_planes_matrix = np.random.normal(size=(num_planes, num_dimensions))` and multiple sets like this `planes = [np.random.normal(size=(num_planes, num_dimensions)) for _ in range(num_sets)]`.<br>
Once you have extracted the vector's neighbors as a subset of all the possible vectors using locality sensitive hashing, you can provide that subset to k-nearest neighbors with k of value 1, so that it will find the most similar word vector among those already neighbouring vectors.

#### Searching documents
K-nearest neighbors can be used to search for text related to a query in a large collection of documents. You simply need to create vectors for both and find the nearest neighbors.

A document can be represented as a vector by summing all the word vectors it contains who are also present in our library of word vectors.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Classification and Vector Spaces](https://www.coursera.org/learn/classification-vector-spaces-in-nlp?specialization=natural-language-processing)
