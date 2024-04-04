# NLP with Sequence Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Sequence Models](#DeepLearningAI-Natural-Language-Processing-Specialization-NLP-with-Sequence-Models)
  - [Week 1: Recurrent Neural Networks for Language Modeling](#Week-1-Recurrent-Neural-Networks-for-Language-Modeling)
    - [Introduction to Neural Networks and Tensorflow](#Introduction-to-Neural-Networks-and-Tensorflow)
      - [Neural Networks for Sentiment Analysis](#Neural-Networks-for-Sentiment-Analysis)  
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

Large [N-Grams](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Probabilistic%20Models#n-grams) are necessary to capture dependencies between distant words. This demands a lot of memory space. RNNs mitigate this issue and outperform N-Gram models in language generation tasks.

##### Recurrent Neural Networks
RNNs are not limited to only taking in account the last N words. They take in account the whole sentence(s) and as a result make better predictions.

RNNs start by computing values with first word. It then propagates those values into calculations with second word to compute other values that will be used during calculations related to third word, and so forth, until it computes values for the to be predicted word. Calculations are repeated for every word in sequence/phrase(s) that is why the neural-network is 'recurrent'. 

Different types of RNN architectures exist. They can take one or many inputs and outputs. For example one image as input and a phrase of multiple words (caption) as output would equal a 'one-to-many' architecture. Other architectures are 'one-to-one', 'many-to-one' and 'many-to-many'. Translation of phrases is an example of a 'many-to-many' architecture. RNNs can thus be implemented for a variety of NLP tasks such as machine translation or caption generation.

##### Math in Simple RNNs
An RNN consists of multiple steps across time t. Each step takes input x<sup>t</sup>, a hidden state h<sup>t-1</sup>, calculates h<sup>t</sup> and makes a prediction ŷ<sup>t</sup>. x<sup>1</sup> may be the first word of a sentence. The hidden state h<sup>t</sup> gets computed with activation function g and subsequently prediction ŷ<sup>t</sup> too.

Here are the activation function formulas:<br>
h<sup>t</sup> = g(W<sub>hh</sub>h<sup>t-1</sup> + W<sub>hx</sub>x<sup>t</sup> + b<sub>h</sub>)<br>
ŷ<sup>t</sup> = g(W<sub>yh</sub>h<sup>t</sup> + b<sub>y</sub>)

You end up training W<sub>hh</sub>, W<sub>hx</sub>, W<sub>yh</sub>, b<sub>h</sub> and b<sub>y</sub>.

##### Cost function for RNN
Using cross-entropy-loss, with K being number of classes, you can caluclate the cost like this:<br>
J = -Σ<sub>i=1</sub><sup>K</sup> y<sub>i</sub> log(ŷ<sub>i</sub>)

RNNs consist of multiple steps t with total amount of steps T. We calculate a ŷ at every step. Thus for RNNs we need to adapt the cross-entropy-loss like this:<br>
J = -1/T Σ<sub>t=1</sub><sup>T</sup> Σ<sup>K</sup><sub>i=1</sub> y<sub>i</sub><sup>t</sup> log(ŷ<sub>i</sub><sup>t</sup>)

For RNNs the loss function is the average loss over its multiple steps.

##### Implement RNNs
Scan functions are like absctract RNNs and allow for faster computation.

A scan function takes a function, list of elements and initialization value. It initializes the hidden state and applies the function on all elements. It basically loops over each step of the RNN.
```
def scan(fn, elems, initializer):
  cur_value = initializer #This variable is same as hidden state H in RNN
  ys = []
  for x in elems:
    y, cur_value = fn(x, cur_value) #This function returns current hidden state H and ŷ similar to the activation function in RNN
    ys.append(y)
  return ys, cur_value
```

The tensorflow framework uses the scan function (`tf.scan()`) as an abstraction that mimics RNNs. Those absctractions are necessary in deep learning frameworks because they allow the use of GPUs and parallel computation for speed.

##### Gated Recurrent Units
Regular RNNs don't work well in a context where word sequences are long. Because the information tends to vanish.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Sequence Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
