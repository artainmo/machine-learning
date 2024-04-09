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
All sequences in LSTM need to be of same size when forming matrices for batch processing. If necessary padding can be used with the associated tag/token \<PAD\>.<br>
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

### Week 3: Siamese Networks
A siamese network consists of two identical neural networks who use the same weights and merge at the end while working alongside two different input vectors to compute comparable output vectors. You can then compare those output vectors to see if they are similar. In NLP, you can use this to identify question duplicates. Platforms like Stack Overflow or Quoara implement such techniques to avoid question duplicates. In NLP, it can also be used to identify similar signatures.

Take the following questions, "How old are you?" and "What is your age?". Those are similar questions even if they are phrased differently. Here, we want to use Siamese Networks to compare the meaning of word sequences and identify question duplicates. We do this by computing a similarity score representing the relationship between the two questions. If that score surpasses a certain treshold we can predict the questions to be of similar meaning and thus duplicates.

While classification learns what an input is, siamese networks learns how similar two inputs are.

#### Architecture
Siamese networks have two identical subnetworks who merge together in the end to produce a final output representing the similarity score. It is important to note that the learned parameters (weights and biases) of each subnetwork are exactly the same.

The following is an example of a Siamese network. Not all Siamese networks are designed to contain LSTMs.<br>
![Screenshot 2024-04-08 at 17 56 59](https://github.com/artainmo/machine-learning/assets/53705599/fb76f738-c1cc-475f-aac9-cfa1a34881c6)<br>
In such a network 'question 1' and 'question 2' represent the inputs. The embedding layer is used to transform the inputs into embeddigs, who are run through an LSTM layer to model the question's meaning. Each LSTM outputs a vector. In the end the cosine similarity is used to measure the similarity between the two output vectors and provides the model's prediction ŷ which will be a value between -1 and 1. If ŷ is greater than some treshold, we will call tau (Τ), then the two questions are deemed similar, else different. Similar questions should output a value closer to 1 while different questions a value closer to -1.

#### Cost function
A siamese network uses a cost function named the 'triplet loss'.

The triplet loss looks at:
* an anchor, which is the base question we will compare another question to.
* a positive, which is another question who is similar to the anchor.
* a negative, which is another question who is different than the anchor.

We expect an anchor and positive to have a cosine similarity close to 1. While we expect an anchor and negative to have a cosine similarity close to -1.<br>
You basically try to minimize the following equation: `cos(A, N) - cos(A, P)`. Where cos refers to cosine similarity, A to anchor, P to positive, and N to negative. If cos(A,P) equals 1 as it should and cos(A,N) equals -1 as it should, then the cost will definitely be less than 0.

The triplet loss uses the margin value alpha to transform slightly negative losses into positive ones to ensure training continues. Else, negative losses can quickly occur and stunt training. For example if alpha is 0.2 and loss -0.1, then we will add alpha to the loss making it positive. It is also important to note that we transform negative losses into 0. So the new loss function becomes `max((cos(A, N) - cos(A, P) + alpha), 0)`.

If you were to select triplets randomly, you would be likely to choose anchors, positives, and negative examples that will lead to losses close to zero. The network will have nothing to learn from those triplets. You can train more efficiently if you choose triplets that create a challenge to the model. So instead of selecting random triplets, you specifically select so called hard triplets where the negative and positive examples are harder to tell apart.

Prepare the data in two batches as a Siamese network takes two inputs. The first batch goes through the first subnetwork of the siamese network to output a matrix consisting of one vector for each batch example. The second batch does the same in second subnetwork.<br>
Each question in batch 1, should be a duplicate of its corresponding question in batch 2 but none of the questions in batch 1 should be duplicates of each other, and the same applies to batch 2. This means for example that the first vector row in batch 1 output matrix is of similar meaning to the first vector row in batch 2 output matrix.<br>
Lastly, we will combine batch 1 output matrix with batch 2 output matrix to form a similarity matrix. The similarity matrix may look like the following.<br>
![Screenshot 2024-04-08 at 22 34 56](https://github.com/artainmo/machine-learning/assets/53705599/57a120db-e8ee-4506-9050-edd17b1ef5aa)<br>
In such a similarity matrix the diagonal should consist of higher values indicating the corresponding questions to be duplicates. Meaning for example, question 3 of batch 1 is similar to question 3 of batch 2. A good functioning model should have higher similarity scores for the duplicates, who lie on the diagonal, compared to the other values who don't represent duplicates.<br>
Creating non-duplicate pairs by using batches like this removes the need for additional non-duplicate examples in the input data. Thus instead of needing specific batches with negative examples, batches with different duplicate pairs can suffice to extract both positive and negative examples. From the similarity matrix, the diagonal values provide duplicate examples and off-diagonal values provide non-duplicate-examples.<br>
The triplet loss function can now be used on the values in the similarity matrix. The overall cost will be the sum of each training example's loss.

The off-diagonal information can be used to improve the triplet loss function. For this we need to extract two values from the off-diagonal values of each row of the similarity matrix.<br>
First, we need to compute the mean negative, which is the mean/average of all the off-diagonal values in each row of the similarity matrix. To clarify, the word 'negative' is used because it refers to non-duplicate-examples which is what the off-diagonal values should represent.<br>
Second, we need to find the closest negative. As mentioned earlier, hard triplets are preferred for training. They have a cosine similarity of the negative example close to the cosine similarity of the positive example. This forces the model to learn the subtle differences they hold. To find the closest negative we search for each row the non-diagonal value who is the closest to, but still less than, the diagonal value of that row.<br>
The triplet loss function can be modified into two losses now.<br>
The first loss looks like this `max((mean_negative - cos(A, P) + alpha), 0)`. We replaced `cos(A, N)` with recently calculated `mean_negative` which helps the model converge faster by reducing noise. The second loss looks like this `max((closest_negative - cos(A, P) + alpha), 0)`. We replaced `cos(A, N)` with recently calculated `closest_negative` which will produce the largest loss and thus highest potential for learning. The final/full loss will be the sum of first and second loss. The overall cost will be the sum of each training example's loss.

#### One Shot Learning
Imagine you want to identify if the author of a certain poem is Lucas or not. You can use a classifier to predict the author of the poem between multiple authors using a large dataset. If the author list stays the same, classification is not a problem. However if the author list changes, the whole model needs to be retrained and the new author needs a solid dataset. Instead, you can use one-shot-learning which consists of comparing one of Lucas' poems to another poem. You can use a learned similarity function for this. Thus, instead of determining the class, we measure the similarity between two classes with one-shot-learning.<br>
One-shot-learning is very useful in banks to identify signatures. When new signatures arrive, instead of retraining the whole classification model, you learn a similarity function that can identify if two signatures are the same or not.<br>
Siamese networks are used to produce a similarity score in one-shot-learning.

#### Train and Test
As explained in [cost function section](#Cost-function), first prepare your data in two batches. The first question in batch 1 is similar to the first question in batch 2, the second one in batch 1 is similar to the second one in batch 2, and so forth. However question one in batch 1 is different than all the other questions in the batch. In general, both in batch 1 and 2, all questions are unique compared to the other questions of the same batch.<br>
Pass those inputs through the subnetworks of the siamese network to get output vectors. Perform cosine similarity on those output vectors to get similarity scores. Those similarity scores can be used to calculate the cost or make a prediction.

When testing the model you will perform one-shot-learning by finding a similarity score between two inputs and comparing it to the treshold value tau to predict if duplicate or not.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Sequence Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
