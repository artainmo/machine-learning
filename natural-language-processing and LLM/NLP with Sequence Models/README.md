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

You understand a word based on your understanding of previous words. Your thoughts have persistence, this is maybe what we call working-memory. Traditional neural-networks cannot persist information but RNNs have loops in them that allow information to persist.

Large [N-Grams](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Probabilistic%20Models#n-grams) are necessary to capture dependencies between distant words. This demands a lot of memory space. RNNs mitigate this issue and outperform N-Gram models in language generation tasks.

##### Recurrent Neural Networks
RNNs are not limited to only taking in account the last N words. They take in account the whole sentence(s) and as a result make better predictions.

RNNs start by computing values with first word. It then propagates those values into calculations with second word to compute other values that will be used during calculations related to third word, and so forth, until it computes values for the to be predicted word. Calculations are repeated for every word in sequence/phrase(s) that is why the neural-network is 'recurrent'. 

Different types of RNN architectures exist. They can take one or many inputs and outputs. For example one image as input and a phrase of multiple words (caption) as output would equal a 'one-to-many' architecture. Other architectures are 'one-to-one', 'many-to-one' and 'many-to-many'. Translation of phrases is an example of a 'many-to-many' architecture. RNNs can thus be implemented for a variety of NLP tasks such as machine translation or caption generation.

###### Math in Simple RNNs
An RNN consists of multiple steps across time t. Each step takes input x<sup>t</sup>, a hidden state h<sup>t-1</sup>, calculates h<sup>t</sup> and makes a prediction ŷ<sup>t</sup>. x<sup>1</sup> may be the first word of a sentence. The hidden state h<sup>t</sup> gets computed with activation function g and subsequently prediction ŷ<sup>t</sup> too.

Here are the activation function formulas:<br>
h<sup>t</sup> = g(W<sub>hh</sub>h<sup>t-1</sup> + W<sub>hx</sub>x<sup>t</sup> + b<sub>h</sub>)<br>
ŷ<sup>t</sup> = g(W<sub>yh</sub>h<sup>t</sup> + b<sub>y</sub>)

You end up training W<sub>hh</sub>, W<sub>hx</sub>, W<sub>yh</sub>, b<sub>h</sub> and b<sub>y</sub>.

###### Cost function for RNN
Using cross-entropy-loss, with K being number of classes, you can caluclate the cost like this:<br>
J = -Σ<sub>i=1</sub><sup>K</sup> y<sub>i</sub> log(ŷ<sub>i</sub>)

RNNs consist of multiple steps t with total amount of steps T. We calculate a ŷ at every step. Thus for RNNs we need to adapt the cross-entropy-loss like this:<br>
J = -1/T Σ<sub>t=1</sub><sup>T</sup> Σ<sup>K</sup><sub>i=1</sub> y<sub>i</sub><sup>t</sup> log(ŷ<sub>i</sub><sup>t</sup>)

For RNNs the loss function is the average loss over its multiple steps.

###### Implement RNNs
Scan functions are like absctract RNNs and allow for faster computation.

A scan function takes a function, list of elements and initialization value. It initializes the hidden state and applies the function on all elements. It basically loops over each step of the RNN.
```
def scan(fn, elems, initializer):
  cur_value = initializer #This variable is same as hidden state h in RNN
  ys = []
  for x in elems:
    y, cur_value = fn(x, cur_value) #This function returns current hidden state h and ŷ similar to the activation function in RNN
    ys.append(y)
  return ys, cur_value
```

The tensorflow framework uses the scan function (`tf.scan()`) as an abstraction that mimics RNNs. Those absctractions are necessary in deep learning frameworks because they allow the use of GPUs and parallel computation for speed.

##### Gated Recurrent Units
Regular RNNs don't work well in a context where word sequences are long. The Gated Recurrent Unit (GRU) is a more complex model able to handle longer word sequences.

One important difference is that GRUs allow relevant information to be kept in the hidden state over long sequences. While information tends to vanish over long sequences in regular RNN models.

Take the text 'Ants are really interesting. ___ are everywhere.' The missing word is 'They' because 'ants' is plural. GRUs remember such important information that occurs at the beginning of text.

GRUs perform additional calculations. They calculate the 'relevance gate' (Γ<sub>r</sub>) and 'update gate' (Γ<sub>u</sub>) at the beginning of each step. These calculate the sigmoid activation function (σ)and thus output a vector with values between 0 and 1. They keep/update relevant information in the hidden state (h<sup>t</sup>).

The relevance gate (Γ<sub>r</sub>) finds a relevance score and allows computing the hidden state candidates (h'<sup>t</sup>).<br>
The hidden state's candidates (h'<sup>t</sup>) stores the information that could be used to override the one passed from the previous hidden state (h<sup>t-1</sup>).<br>
The updates gate (Γ<sub>u</sub>) determines how much information from the previous hidden state (h<sup>t-1</sup>) will be overwritten and allows computing the current hidden state (h<sup>t</sup>).<br>
The final prediction (ŷ) is calculated using the current hidden state (h<sup>t</sup>) similarly as in a regular RNN.

What follows are the mathematical formulas. `[x, y]` in mathematical notation indicates concatenation between matrices.<br>
Γ<sub>u</sub> = σ(W<sub>u</sub>[h<sup>t-1</sup>, x<sup>t</sup>] + b<sub>u</sub>)<br>
Γ<sub>r</sub> = σ(W<sub>r</sub>[h<sup>t-1</sup>, x<sup>t</sup>] + b<sub>r</sub>)<br>
h'<sup>t</sup> = tanh(W<sub>h</sub>[Γ<sub>r</sub> * h<sup>t-1</sup>, x<sup>t</sup>] + b<sub>h</sub>)<br>
h<sup>t</sup> = (1 - Γ<sub>u</sub>) * h<sup>t-1</sup> + Γ<sub>u</sub> * h'<sup>t</sup><br>
ŷ<sup>t</sup> = g(W<sub>y</sub>h<sup>t</sup> + b<sub>y</sub>)

##### Deep and Bi-directional RNNs
Deep RNNs allow the capture of more dependencies and thus make better predictions. Deep RNNs are multiple regular RNNs stacked together. Each regular RNN inside a deep RNN is considered as a layer. Each layer calculates its hidden states and passes its activations to the next layer.

Take the following text example 'I was trying really hard to get a hold of ___. Louise, finally answered when I was about to give up.'. Regular RNNs will only read what comes before the word that has to be predicted and as a result cannot correctly predict 'Louise' in this example. Bi-directional RNNs also read from the end to the beginning thus they also read what comes after the word that has to be predicted. In bi-directional RNN, to make a prediction ŷ, hidden states from both directions are combined to form one hidden state. Bidirectional RNNs are acyclic graphs, which means that the computations in one direction are independent from the ones in the other direction.

### Week 2: LSTMs and Named Entity Recognition
Named entity recognition (NER) is a subtask of information extraction that locates and classifies named entities. Named entities can refer to organizations, persons, locations. For example, if you look at the sentence 'The French people are visiting Morocco for Christmas.', Fench is a geopolitical entity, Morocco is a geographic entity and Christmas is a time indicator.

To implement NER we will use long short-term memory units (LSTMs). They are similar to GRUs except that they have even more gates.

#### RNNs and Vanishing Gradients
Vanishing or exploding gradients are problems commonly found in RNNs who deal with long sequences. They consist of values used during backpropagation calculations coming close to zero or infinity respectively. If coming close to zero, certain items' contributions get neglected, and if approaching infinity, convergence problems arise. 

Initializing weights to an identity matrix and using the ReLU activation function are ways of preventing vanishing gradients. Skip connections consists of direct connections to earlier layers, thus skipping activation functions and preventing those earlier values from vanishing through the layers, thus allowing them to impact the final output.<br>
Gradient clipping consists of transformation any value larger than a predefined value to that predefined value. This helps prevent exploding gradients.

GRUs and LSTMs mitigate those problems.

#### Introduction to LSTMs
LSTM is a variety of RNN that allows your model to remember and forget certain inputs. It is composed of a cell state which represents its memory, a hidden state to compute what changes to make, and lastly three gates that transform the states in the network to remember and forget informations. These gates also prevent the risk of vanishing or exploding gradients across long sequences.<br>
LSTMs are basically useful to handle long sequences by remembering important past information while preventing gradient problems.

An LSTM stores information in the cell state and hidden state, who are denoted by c and h respectively. Inputs are denoted by x and outputs by ŷ.<br>
In an LSTM, multiple computations are performed in a single unit. The information flows through three different gates:
* First, the forget gate, which uses the inputs and previous hidden state to decide what information from the cell state to forget.
* Second, the input gate, decides what information from the inputs and previous hidden state should be remembered and thus added to the cell state.
* Lastly, the output gate determines what information from the cell state gets stored in the hidden state and used to construct an output.

LSTMs are useful for building language models. It can be used to predict the next character in your email, build chatbots capable of remembering longer conversations. Music is composed of long sequences of nodes like text uses long word sequences, and thus LSTMs can also compose music. Other applications are automatic image captioning and speech recognition.

#### LSTM architecture
The three gates start by using the sigmoid activation function on the input and previous hidden state to ensure values are between 0 and 1. Because gates use such values to indicate the degree to which information can flow, with a value of 0 meaning the gate is closed and does not let information through while a value of 1 lets the information flow through freely.

The candidate cell state is another important computation made. It starts by using the tanh activation function on input and previous hidden state to squeeze values between -1 and 1. This transformation is used to improve training performance by preventing any of the values from the current inputs from becoming so large that they make the other values insignificant.<br>
Once you have the forget gate, input gate and the candidate cell state, you can update the cell state. To compute the new cell state you need to sum the current cell state passing through the forget gate with the candidate cell state passing through the input gate.<br>
Finally, you can compute the new hidden state used to produce the output. For that, you can optionally first pass the new cell state through the tanh activation function. Subsequently through the output gate.

#### Introduction to Named Entity Recognition
Many NLP systems make use of an NER component to handle named entities. 

Different entity types exist such as:
* geographical entities - ex. Thailand
* organization entities - ex. Google
* geopolitical entities - ex. Indian
* time indicator entities - ex. December
* artifact entities - ex. artifact
* person entities - ex. Barack Obama

Take the phrase 'Sharon flew to Miami last Friday'. A NER system would classify Sharon as a personal name, Miami as a geographical entity, and Friday as a time indicator.

NERs are used to improve search engine efficiency, recommendation engines, matching customer to proper customer service, and in automatic trading it can help find relevant news articles, subsequently using sentiment analysis on those news articles. NERs help by scanning large documents for certain tags quickly. In the case of a search engine the queried tags can be matched against tags in documents. Recommendation engines would for example scan tags in your history against matching tags for new candidate material.

#### Training NERs: Data Processing
First, assign each entity class/type a unique number. Also convert each word in text into its associated number in an array. Basically using an integer representation for both text input and entity classes.<br>
All sequences in LSTM need to be of same size. If necessary padding can be used with the associated tag/token \<PAD\>.<br>
Create a data generator to output the created tensors (vectors/matrices of numbers) in batches which speeds up training.<br>
Then, feed the batches into an LSTM unit. Its output will be run trough a Dense layer and a prediction is made using log-softmax over K classes.

Here is how layers might look like in practice:
```
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(),
    tf.keras.layers.LSTM(),
    tf.keras.layers.Dense()
])
```

#### Computing accuracy
After taining the NER you need to evaluate it.

Make predictions on test set using your model. Take the highest value in prediction array which represents the highest probability entity class and thus made prediction. Those predictions can be compared against true labels to see how accurate the model is on unseen data.

Padded tokens /<PAD/> need to be masked/skipped when calculating accuracy.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Sequence Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
