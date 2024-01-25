# natural-language-processing and LLM
## Table of contents
- [DeepLearning.AI course: Natural Language Processing Specialization](#deeplearningai-course-natural-language-processing-specialization)
- [Resources](#Resources)

## DeepLearning.AI course: Natural Language Processing Specialization
### Week 1: Sentiment Analysis with logistic regression
In the first week we will learn about logistic regression. Logistic regression takes as inputs features (X) and labels (Y). It uses parameters (theta) to make predictions (Yhat). While training, parameters are updated in relation to the error (cost) in Yhat compared to Y, until the cost becomes minimized.

In example exercise we will take a tweet and predict if it has a positive or negative sentiment. Positive sentiment tweets will have a label Y of 1 and negative sentiment tweets a label Y of 0. First we will need to extract features, train model and lastly classify the tweets by making predictions with trained model.

To extract features, text will have to get encoded as an array of numbers. For that, first you need to create a vocabulary list that consists of all the unique words in the read/learned text(s). This vocabulary list will represent a list of all the features (X). For the example text to learn/predict on replace each word in vocabulary list that is present in that text with 1 else 0. This is how we give the correct inputs to the features.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)
* [codecademy - Apply Natural Language Processing with Python](https://www.codecademy.com/learn/paths/natural-language-processing)
