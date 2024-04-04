# NLP with Sequence Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Sequence Models](#DeepLearningAI-Natural-Language-Processing-Specialization-NLP-with-Sequence-Models)
  - [Week 1: Recurrent Neural Networks for Language Modeling](#Week-1-Recurrent-Neural-Networks-for-Language-Modeling)
    - [Introduction to Neural Networks and Tensorflow](#Introduction-to-Neural-Networks-and-Tensorflow)
      - [Introduction](#Introduction)  
- [Resources](#Resources)

## DeepLearning.AI: Natural Language Processing Specialization: NLP with Sequence Models
### Week 1: Recurrent Neural Networks for Language Modeling
#### Introduction to Neural Networks and Tensorflow
Logistic regression and naive bayes were used during first NLP course for sentiment analysis. In this course we will use deep neural networks to perform sentiment analysis more robustly.

The first lesson of this week is here for you to remember how neural networks work and to revisit the problem of classification that you have already seen in previous courses. The sequential models start in the second lesson of this week. The first week helps you transition from traditional NLP techniques in the previous courses to the sequence-based models in this course.

##### Neural Networks for Sentiment Analysis
We will perform sentiment analysis on tweets using a neural network with two hidden layers. The neural network will take as input a simple vector representation of a tweet. The first hidden layer will be an embedding layer that transforms the input vector into an optimal representation of the tweet. The second hidden layer uses a ReLU activation function and output layer uses softmax activation function to predict if tweet has positive or negative sentiment.

To create the input vector, we use an integer representation. First, take the vocabulary and assign each word an incrementing index that starts at 1. The tweet's words will be replaced by associated integers to form a vector. Take the longest tweet/vector and now make all other vectors have the same length by adding zeros to them, a process we call padding.

##### Different Layers
We can make distinction between two commonly used layers in neural-networks. Namely, dense and ReLU layers.

The dense layer refers to the synapses between nodes who contain weights. In this layer the dot product between input matrix and weight matrix is calculated.<br>
The ReLU layer is the actual node layer who receives as input the dense layer's output and who performs the activation function ReLU on that input.

Embedding and mean layers also exist. In serial models the mean layer follows the embedding layer.

An embedding layer consists of trainable weights who represent the word embeddings. Word embeddings, being vectors holding the meaning of words numerically. Its row length is the vocabulary's length and column length is embedding length. Thus each row vector can represent one vocabulary word.

When padded vectors come into the embedding layer, certain rows of the embedding layer will equal zero. To avoid those zero values we use the mean layer to transform the weight matrix found in embedding layer into a one dimensional vector of embedding length that takes the average of each embedding layer column, thus eliminating the rows consisting of zero values. This layer has no trainable parameters as it is only calculating the mean of word embeddings.

#### N-Grams vs Sequence Models
Sequence models are a class of machine learning models designed for tasks that involve sequential data, where the order of elements in the input is important. Sequential data includes textual data, time series data, audio signals, video streams or any other ordered data.<br>
A recurrent neural network (RNN) can model sequence data and thus form a sequence model.

Large [N-Grams](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Probabilistic%20Models#n-grams) are necessary to capture dependencies between distant words. This demands a lot of memory space. RNNs solve this issue and outperform N-Gram models in language generation tasks.

##### Recurrent Neural Networks
RNNs are not limited to only taking into account the last N words. They take in account the whole sentence and as a result make better predictions.



## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Sequence Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
