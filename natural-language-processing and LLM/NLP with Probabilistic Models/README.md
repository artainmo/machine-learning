# NLP with Probabilistic Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Probabilistic Models](#deeplearningai-natural-language-processing-specialization-nlp-with-probabilistic-models)
  - [Week 1: Autocorrect](#Week-1-Autocorrect)
    - [Introduction](#introduction)
    - [Building the model](#Building-the-model)
    - [Minimum edit distance](#Minimum-edit-distance)
    - [Minimum edit distance algorithm](#Minimum-edit-distance-algorithm)
  - [Week 2: Part of Speech Tagging and Hidden Markov Models](#week-2-part-of-speech-tagging-and-hidden-markov-models)
    - [Part of Speech Tagging](#Part-of-Speech-Tagging)
    - [Markov chains](#Markov-chains)
    - [Hidden Markov Models](#Hidden-Markov-Models)
    - [Calculating probabilities](#Calculating-probabilities)
    - [The Viterbi Algorithm](#The-Viterbi-Algorithm)
      - [Viterbi: Initialization](#Viterbi-Initialization)
      - [Viterbi: Forward pass](#Viterbi-Forward-pass)
      - [Viterbi: Backward pass](#Viterbi-Backward-pass)
  - [Week 3: Autocomplete and language models](#week-3-autocomplete-and-language-models)
    - [N-Grams](#n-grams)
    - [Sequence Probabilities](#Sequence-Probabilities)
    - [The N-Gram Language Model](#the-n-gram-language-model)
    - [Language Model Evaluation](#language-model-evaluation)
    - [Out of vocabulary words](#out-of-vocabulary-words)
    - [Missing N-Grams](#Missing-N-Grams)
  - [Week 4: Word embeddings with neural networks](#week-4-word-embeddings-with-neural-networks)
    - [Introduction](#introduction-1)
    - [Word representations](#Word-representations)
    - [Word embedding methods](#Word-embedding-methods)
    - [Continuous Bag-of-Words Model](#continuous-bag-of-words-model)
      - [Data preparation](#Data-preparation)
      - [Architecture](#Architecture)
      - [Cost function](#Cost-function)
      - [Forward and backward propagation](#Forward-and-backward-propagation)
      - [Extracting word embedding vectors](#Extracting-word-embedding-vectors)
    - [Evaluating word embeddings](#Evaluating-word-embeddings)
- [Resources](#Resources)

## DeepLearning.AI: Natural Language Processing Specialization: NLP with Probabilistic Models
### Week 1: Autocorrect
#### Introduction
In this course we will learn about word models and how to use them to predict word sequences. This allows autocorrection as well as web search suggestions. 

In the first week we will build an autocorrect system by using probabilities of character sequences. An important concept is 'minimum edit distance' which consists of evaluating the minimum amount of edits to change one word into another. 'Dynamic programming' is the 'minimum edit distance' algorithm we will use. It is an important programming concept which frequently comes up in interviews and could be used to solve a lot of optimization problems.

Autocorrect is an application that changes misspelled words into the correct ones. It works first by identifying misspelled words, second find words who are a certain amount of edits away, lastly calculate word probabilities which calculates the chance of a word appearing in a certain context.<br>
In this week's exercise we won't take the following in account, but autocorrect can also change words who exist but are used in the wrong context. For example "Happy birthday deer friend", where 'deer' is an existing word but should be replaced with 'dear' in this context.

#### Building the model
We can identify misspelled words by verifying if they exist in a dictionary.<br>
Once a word is misspelled we can start generating strings who are a certain amount of edits away from the misspelled word. You can do that by inserting a letter, removing a letter, swap two adjacent letters or replace one letter with another. A lot of those generated strings won't be actual words, thus we need to filter out those non-existing words by verifying if they exist in our dictionary.

Now that we have a list of words we can calculate their probabilities. We can start by creating a frequency dictionary. Which consist of a dictionary containing words as keys and probability of a word (calculated by dividing word frequency in text by total amount of words in text) as values. The word with highest probability will be used for autocorrection. If you want to build a slightly more sophisticated auto-correct you can keep track of two words occurring next to each other instead. For this week however we will be implementing the probabilities by just using the word frequencies.

#### Minimum edit distance
Minimum edit distance can evaluate the similarity between two strings. It represents the number of edits needed to transform one string into another.

We make distinction between three edit operations:
* Insert (add a letter)
* Delete (remove a letter)
* Replace (change 1 letter to another)

'Insert' has an edit cost of 1, 'delete' too and 'replace' has an edit cost of 2. Thus if 'play' needs two replace edits to become 'stay', the total edit cost will equal 2 + 2 = 4. Measuring the edit distance like this is known as Levenshtein distance.

As strings become larger, it becomes harder to calculate the 'minimum edit distance'. This is why we will use the 'minimum edit distance' algorithm called 'dynamic programming'.

#### Minimum edit distance algorithm
First we will create a distance matrix called D.

|   | # | s | t | a | y |
| - | - | - | - | - | - |
| # | 0 | 1 | 2 | 3 | 4 |
| p | 1 | 2 | 3 | 4 | 5 |
| l | 2 | 3 | 4 | 5 | 6 |
| a | 3 | 4 | 5 | 4 | 5 |
| y | 4 | 5 | 6 | 5 | 4 |

The # indicates an empty string. The minimum edit distance between two empty strings is 0. The minimum edit distance between p or s and an empty string is 1. The minimum edit distance between p and s is 2.<br>
For the transformation of pl into an empty string the minimum edit distance is 2, for the transformation of pla into an empty string the minimum edit distance is 3 and for the transformation of play into an empty string the minimum edit distance is 4.<br>

To calculate all the cells, three equations can be used, depending on what edit operation you should use to find the minimum amount of edits:
* `D[i,j] = D[i-1,j] + delete cost`. This indicates you want to populate the current cell (i,j) by using the cost in the cell found directly above. Delete cost is usually equal to 1.
* `D[i,j] = D[i,j-1] + insert cost`. This indicates you want to populate the current cell (i,j) by using the cost in the cell found directly to its left. Insert cost is usually equal to 1.
* `D[i,j] = D[i-1,j-1] + (replace cost if source[i] != target[j] else 0)`. Replace cost is usually equal to 2.

D[m,n], in above example D[4,4] equaling 4, represents the minimum edit distance between the two compared words, in this example 'stay' and 'play'.

Sometimes the minimum edit distance is not sufficient. You need to know the steps to get it. The distance matrix D remembers those steps by calculating them all. A backtrace lets you know what path you took across the distance matrix from the top left to bottom right corner.

Dynamic programming refers to solving a small problem to help solve a bigger problem, to then solve an even bigger problem using prior result, and so forth. This is what we did in the distance matrix D, we started calculating the edit distance of two empty strings and ended with calculating the edit distance between two complete words.<br>
Dynamic programming is not only used for minimum edit distance, it can also be used to find the shortest paths as in Google maps.

### Week 2: Part of Speech Tagging and Hidden Markov Models
#### Part of Speech Tagging
Part of speech (POS) refers to categories of words such as noun, verb, adjective, adverb, pronoun, preposition. Tags are used to tag the words in a sentence with their part of speech category.

Semantics is the study of meaning in language. It can be applied to entire texts or to single words. For example, "destination" and "last stop" technically mean the same thing, but students of semantics analyze their subtle shades of meaning.<br>
Because POS tags describe the characteristic structure of words in a sentence or text, you can use them to make assumptions about semantics. Named entities can be identified using POS tags. 

Co-reference resolution consists of determining what a pronoun refers to. For example take the following text: 'The Eiffel tower is beautiful. It is found in Paris'. Here the pronoun 'it' refers to the named entity 'Eiffel tower'. POS tags are used for co-reference resolution.

When guessing/suggesting words you can use the probabilities of POS tags occuring near one another to come up with the most reasonable output. This is used in speech recognition.

#### Markov chains
Markov chains are used for identifying the probability of the next word using POS tagging.

We can predict the POS tag of a word in a sentence from the POS tag of the previous word in the sentence. Each POS tag has a certain chance of being followed by a certain POS tag.<br>
A markov chain can be depicted as a directed graph. A graph wherein each node represents a POS tag connected to another POS tag node by a one direction line representing the probability that the initial POS tag is followed by the POS tag it is connected to.<br>
We can call those graph nodes, the 'states' of our model. `Q = {q1, q2, q3}` is a way of mathematically describing/writing the set of states in our model.<br>
Transition probabilities describe the chance of one POS tag being followed by another POS tag, thus they are the same as the directed lines in our graph.

Markov chains can also be represented in a table that we would call a transition matrix.
|      | noun | verb |
| ---- | ---- | ---- |
| Init | 0.7  | 0.3  |
| noun | 0.4  | 0.6  |
| verb | 0.6  | 0.4  |

In such a matrix the first row represents the current states, first column the potential next states, and the values inside the matrix represent the associated transiton probabilities.<br>
The 'init' state refers to initial and is used for predicting the first word of a sentence, thus the initial word that cannot be predicted from a previous word.<br>
Such a matrix can be mathematically written/described like this for example:
```
    |a1,1  a1,2  a1,3|
A = |a2,1  a2,2  a2,3|
    |a3,1  a3,2  a3,3|
```

#### Hidden Markov Models
Hidden Markov models are used to decode the hidden state of a word. In this case, the hidden state is the POS tag of that word.<br>
The POS tag of a word is considered hidden because when reading the text a machine cannot automatically deduce the POS of each word in that text.<br>

In hidden Markov models, emission probabilities are used to give the probability of one state (POS tag) going to a specific word.<br>
A hidden Markov model can be descibed inside an emission matrix. For example:
|      | lamp | pear |
| ---- | ---- | ---- |
| verb | 0.7  | 0.3  |
| noun | 0.4  | 0.6  |

Such a matrix can be mathematically written/described like this for example:
```
    |b1,1  b1,2  b1,3|
B = |b2,1  b2,2  b2,3|
    |b3,1  b3,2  b3,3|
```

This emission matrix B, will be used together with the transition matrix A, to help identify the POS tag of a word in a sentence.

#### Calculating probabilities
Given a corpus, we can calculate the probabilities that populate our transition and emission matrices.

To calculate the transition probability of POS tag Alpha to POS tag Beta. First take all the occurences in corpus of Alpha followed by Beta, and divide it by all the occurences of Alpha. Do this for all the POS tag combinations to fill the transition matrix.<br>
To fill the 'init' row you simply need to count the occurence of a POS tag appearing as first word of a phrase divided by total amount of first words.<br>
A division by 0 can occur when a POS tag does not appear in the corpus leading to a probability of 'undefined'. Or a probability of 0 can occur when a POS tag combination does not appear in corpus. This can be problematic when trying to generalize the model. To resolve this problem add a small value 'epsilon' (ex. 0.001) to the counts in the numerator and add 'amount_of_POS_tags * epsilon' to the divisor. This operation can be called smoothing, it prevents 0 values during calculations. Usually smoothing is not used however on the first 'init' row.

To calculate the emission probability, you first need to calculate the number of co-occurence between a POS tag with a specific word, and divide that number by the total amount of occurences of that same POS tag. Do this for all the POS tag, word combinations inside the emission matrix. Similarly we can apply smoothing for generalization of the model.

#### The Viterbi Algorithm
The Viterbi algorithm tries to find the sequence of hidden states, here in other words, the POS tag for each word in a phrase, with the highest probability.<br>
The Viterbi algorithm is a graph algorithm at its core, however matrix manipulation formulas will be used for its calculations.

Let's take the phrase 'I love to learn' with the following graph.<br>
![Screenshot 2024-03-13 at 11 57 11](https://github.com/artainmo/machine-learning/assets/53705599/e822ce54-9faf-4753-8444-ab5c8e6c5972)<br>
The first word 'I' can only be emitted by the O state. As shown in picture the transition probability of initial state (also called Pi) to O state is 0.3 and the emission state from O to 'I' is 0.5. The joint probability is 0.15.<br>
Next the word love can be found in NN and VB states. The probability of prior state O to NN is 0.5 and from O to VB also 0.5. The probability of NN state to word 'love' is 0.1 while of VB state is 0.5. Thus the joint probability of O to VB to 'love' is higher with a value of 0.25. Thus VB will be the chosen POS tag this time.<br>
Word 'to' can only be found in O state. Transition probability from VB state to O state is 0.2 and emission probability from O state to word 'to' is 0.4, thus joint probability is 0.08.<br>
Word 'learn' can only be found in VB state. Transition probability from O state to VB state is 0.5 and emission probability from VB state to word 'learn' is 0.2, thus joint probability is 0.1.<br>
Finally, the probability of this word sequence can be calculated like this, 0.15 * 0.25 * 0.08 * 0.1, and equals 0.0003.

The Viterbi algorithm can be divided into three steps.
1. Initialization
2. Forward pass
3. Backward pass

For our calculations we will use the auxiliary matrices C and D. The matrix C holds the probabilities and matrix D the indices of the visited states.<br>
These two matrices have n rows, where n is the number of POS tags or hidden states in our model, and k columns, where k is the number of words in the given sequence.

##### Viterbi: Initialization
In the initialization step, the first column of each of our matrices C and D is populated.

The first column of C represents the joint probabilities from our initial state (pi) to the POS tags of associated rows and word of associated column. This can be calculated by the product of transition probability from pi to state with emission probability from state to word.

The first column of D is set to 0 as there are no preceding POS tags we have traversed.

##### Viterbi: Forward pass
During the forward pass we will populate the remaining C and D matrix entries, column per column.

To find the probabilities in the C matrix, we first need to take the previous POS tag with highest probability. This can be done by looking at highest probability of previous column. Then we need to multiply the transition probability from prior POS tag to current POS tag with emission probability from current POS tag to word. And finally you need to multiply that by the highest probability of previously traversed path, thus the probability you found to find most probable previous POS tag. The formula looks like this: c<sub>i,j</sub> = max<sub>k</sub>c<sub>k,j-1</sub> * a<sub>k,i</sub> * b<sub>i,cindex(wj)</sub>.

Note that the only difference between C and D, is that in the former you compute the probability and in the latter you keep track of the index of the row where that probability came from. So you keep track of which k was used to get that max probability.

##### Viterbi: Backward pass
Here we will use the previously created matrices C and D to create a path and assign a POS tag to every word.

We have to extract the path through our graph using the matrix D. First we go to last column of matrix C and there take the index of the highest probability row. We call this index s and use the formula: s = argmax<sub>i</sub>c<sub>i,K</sub>. Then go back to D, look at last column and the value at index s. That value equates the row index of previous column. Thus for example if s equals 1, we know our last state to be state 1. If the value inside D last row at that index 1 is 3, we know the second last state to be 3. Then we need to go to second last column of D at index 3 and find the value that will tell us the third last state. And so forth. Look at following example of traversing the matrix D.<br>
![Screenshot 2024-03-13 at 13 38 07](https://github.com/artainmo/machine-learning/assets/53705599/4e0cfcb2-49c3-42e7-8c12-9358cbc35b8e)<br>
This thus gives us the sequence of states (POS tags) for our sequence of words.

Coution with indices as in python they start with 0 not 1.<br>
Also caution when using probabilities with very small values. To avoid very small values use 'log probabilities'. With log probabilities we addition the previous probability with logarithm of transition probability and the logarithm of emission probability when calculating matrix C, instead of multiplying the previous probability, transition and emission probabilities.

### Week 3: Autocomplete and language models
#### N-Grams
N-Grams are fundamental in NLP and foundational for understanding more complicated models.<br>
This week we will create an N-Gram language model from a text corpus and use it to auto-complete a sentence. A corpus can be any text collection but generally is a large database of text documents, such as all the Wikipedia pages, or books from one author, or tweets from one account. A language model estimates the probability of an upcoming word given a history of previous words.<br>
An N-Gram language model is created from a text corpus. Users of the autocomplete system will provide the starting words and the model should answer by predicting and suggesting the following words.<br>
N-Gram language models can be used in speech recognition to predict what got most likely heard. They can also be used in spelling correction to identify the use of incorrect words in a phrase's context. Search suggestion tools also use N-Gram language models.

An N-Gram is a sequence of unique words wherein the order matters. When processing the corpus, punctuations are treated like words, but other special characters are removed.<br>
Take the corpus 'I am happy because I am learning'. A unigram is a set of all unique words. The unigram for prior text corpus would be {I, am , happy, because, learning}. A bigram is a set of all unique two words combinations that appear side-by-side. Here the bigram would be {I am, am happy, happy because, because I, am learning}. Trigrams represent unique triplets of words that appear together in the corpus.

A corpus consists of a sequence of words and that sequence can be denoted as 'w<sub>1</sub><sup>m</sup> = w<sub>1</sub> w<sub>2</sub> ... w<sub>m</sub>'. To denote only a subsequence of that vocabulary 'w<sub>1</sub><sup>3</sup> = w<sub>1</sub> w<sub>2</sub> w<sub>3</sub>'.

The probability of a unigram can be calculated by dividing the total amount of occurences of a word by the total amount of words in the corpus. The probability of a word B occuring if the previous word was A can be calculated by dividing the total amount of AB bigrams by the total amount of A unigrams. The probability of the word C occuring next to words AB can be calculated by dividing the total amount of ABC trigrams by the total amount of AB bigrams.<br>
The N-Gram probability formula can be written as such 'P(w<sub>N</sub> | w<sub>1</sub><sup>N - 1</sup>) = c(w<sub>1</sub><sup>N - 1</sup>w<sub>N</sub>) / C(w<sub>1</sub><sup>N - 1</sup>)'.

#### Sequence Probabilities
To calculate the probability of the following phrase 'the teacher drinks tea' we need to multiply the probability of 'the' with the probability of 'the' being followed by 'teacher' and with the probability of 'the teacher' being followed by 'drinks' and with the probability of 'the teacher drinks' being followed by 'tea'.<br>
Mathematically we can write it like this: P(the teacher drinks tea) = P(the)P(teacher|the)P(drinks|the teacher)P(tea|the teacher drinks).<br>
As sentences become longer, the chance of it occuring elsewhere in the corpus becomes smaller and smaller. This leads to probability calculations with zero values which is erroneous.<br>
Thus instead you may want to approximate the probability result by only using probabilities of two words occuring together (bigrams). This would look like: P(the teacher drinks tea) ≈ P(teacher|the)P(drinks|teacher)P(tea|drinks). This is based on the Markov assumption which states that the probability of each word only depends on N previous words.

If conditional probabilities used in N-Grams are calculated using a sliding window of two or more words, what happens at the beginning and end of a sentence?<br>
The start symbol \<s\> is used to indicate the beginning of a sentence and is used to calculate the bigram probability of the sentence's first word. This means 'the teacher drinks tea' becomes '\<s\> the teacher drinks tea' and P(\<s\> the teacher drinks tea) ≈ P(the|\<s\>)P(teacher|the)P(drinks|teacher)P(tea|drinks). For trigrams you would use two '\<s\>' start symbols for the first word and one for second word.<br>
The end symbol '\</s\>' is used at end of phrase to calculate the bigram probability of the sentence's last word like this: P(\<s\> the teacher drinks tea \</s\>) ≈ P(the|\<s\>)P(teacher|the)P(drinks|teacher)P(tea|drinks)P(\</s\>|tea). You always only need one end symbol.

#### The N-Gram Language Model
A count matrix indicates how many times each word is followed by every other word in a corpus. Thus it captures N-Gram occurences. Next the count matrix will be transformed into a probability matrix indicating the conditional probabilities of the N-Grams.

A bigram count matrix of corpus '\<s\>I study I learn\</s\>' looks like this.
|       | \<s\>| \</s\>| I | study | learn |   | *sum*   |
| ----- | ---  | ----  | - | ----- | ----- | - | ------- |
| \<s\> | 0    | 0     | 1 | 0     | 0     |   | ***1*** |
| \<\s\>| 0    | 0     | 0 | 0     | 0     |   | ***0*** |
| I     | 0    | 0     | 0 | 1     | 1     |   | ***2*** |
| study | 0    | 0     | 1 | 0     | 0     |   | ***1*** |
| learn | 0    | 1     | 0 | 0     | 0     |   | ***1*** |


We can transform it into a probability matrix by dividing the cell contents by the row sums.
|       | \<s\>| \</s\>| I | study | learn |
| ----- | ---  | ----  | - | ----- | ----- |
| \<s\> | 0    | 0     | 1 | 0     | 0     |
| \<\s\>| 0    | 0     | 0 | 0     | 0     |
| I     | 0    | 0     | 0 | 0.5   | 0.5   |
| study | 0    | 0     | 1 | 0     | 0     |
| learn | 0    | 1     | 0 | 0     | 0     |

The language model will now consist of a script that uses the created probability matrix to estimate word sequence probabilities. It splits sentences into N-Grams and estimates their probabilities using the probability matrix. By extracting the last word of the highest probability N-Gram it can predict the next word.

When multiplying probabilities who consist of small values we are at risk of numerical underflow. This is why we can use 'log probabilities' which consists of replacing multiplications of numbers with additions of logarithms of those numbers.

Generative Language Models generate text from scratch and make use of N-Gram language models. They start by choosing a bigram starting with starting symbol \<s\> and continue by choosing the highest probability N-Grams that start with previous word to generate text until encountering the ending symbol \<\s\>.

#### Language Model Evaluation
First, split the corpus into a train (80%), validation (10%) and test (10%) set. If the corpus is very large, the validation and test set can each only be 1% of the total corpus. We can split on continuous text or on random short word sequences.

The perplexity metric is used to evaluate the test set. It measures the complexity of text samples. Human written text is more likely to have a lower perplexity score. On the other hand a text generated by random word choices would be more complex.<br>
The perplexity is measured by computing the probabilies of all sentences in test set and raise that probability to the power of -1 over m (amount of words, not counting starting symbols but counting ending symbol). Mathmatically it is written like this: PP(W) = P(s<sub>1</sub>, s<sub>2</sub>, ..., s<sub>m</sub>)<sup>-1/m</sup>. So the higher the probability estimate, the lower the perplexity. Thus smaller perplexity is indicative of a better model.

To calculate the perplexity of a bigram model, you first need to get the probability of all sentences in test set by calculating product of bigram probabilities of all sentences. Then take that to the power of -1 over m. Sometimes 'log perplexity' is used whereby -1/m is multiplied by the sum of logarithms with base 2 of all bigram probabilities. Again, this would be used to prevent underflow.

#### Out of vocabulary words
A vocabulary is a set of unique words supported by our language model. Sometimes you may encounter words who are not part of this vocabulary. We can call those, unknown words, or out of vocabulary words (OOV). To handle OOV you can replace them by the special tag \<UNK\> in corpus and calculate your probabilities by treating \<UNK\> as any other word.

Criterias can exist for words of the corpus to become part of the vocabulary. For example a word may need to appear a minimum amount of times in the corpus to become part of the vocabulary. You can also define a maximum vocabulary size and select words with highest frequency to be part of that limited vocabulary. 

\<UNK\> tends to lower perplexity score. Thus using \<UNK\> excessively can lead to false belief of good model. Thus use \<UNK\> sparingly. When using the perplexity metric, only language models with same vocabulary are comparable.

#### Missing N-Grams
If certain N-Grams are missing in the corpus their probability will result in 0. Laplacian smoothing, also called add-one smoothing, can be used to avoid N-Gram probabilities equaling 0. To use Laplacian smoothing when calculating N-Gram probabilities, add 1 to the numerator and add the vocabulary size to the denominator. With add-k-smoothing, add k to the numerator and add the vocabulary size times k to the denominator.

Another way of dealing with missing N-Grams is to use the Backoff method. Which consists of lowering the N order, use the associated (N - 1)-Gram, until the N-Gram is not missing anymore. Thus if for example you search probability of trigram 'are you happy' but you cannot find a probability for it or the given probability is 0, then look at the associated bigram 'you happy' and if its probability isn't found either, you can search the probability of unigram 'happy'. If using the probability of a lower-N-gram you need to multiply that probability with lambda (value between 0 and 1) times the difference in N.<br>
Alternatively, the interpolation method can be used. It consists of combining the probability of the N-Gram with the (N - 1)-Gram probability down to the Uni-Gram probability. For example you would calculate the following trigram like this: P(chocolate|John drinks) = 0.7 x P(chocolate|John drinks) + 0.2 x P(chocolate|drinks) + 0.1 x P(chocolate). As you can see more weight is given to higher order N-Grams but lower order N-Grams are still taken in account to avoid 0 values. Those 'weights' are called lambdas and when summed need to equal 1.

### Week 4: Word embeddings with neural networks
#### Introduction
Word embeddings numerically represent the meaning of words via vectors. They are used in most NLP applications as they allow the transformation of text into numerical code.

Machine learning models learn the meaning of words by creating word embeddings. Although we will learn to implement such basic models from scratch, in the real world NLP libraries can be used to easily implement more advanced models. Keras and Pytorch are two libraries that easily allow the implementation of word embeddings created from neural networks.

#### Word representations
Imagine a vocabulary of 1000 words. The first word could receive code number 1 and last one code number 1000. This simple integer representation uses an order with no semantic logic. Instead, we can create word vectors consisting of 1000 values all equal to zero besides the value at index of word that can be set to one. Thus, the first word would consist of a vector starting with value one followed by 999 zeros, for example. Those vectors are called one-hot-vectors. Their advantage is that they don't imply any relationship between different words. However, limitations are that they require high memory space and that they don't carry the word's meaning.<br>
Alternatively, word vectors can consist of values representing the degree of attributes for the word. If a word has a positive value of 1.3 and an abstract value of -3, then a word vector can be formed like this (1.3, -3). This type of word vector we call word embeddings. It encodes the word's meaning in a low dimensional space. However, certain words may end up with similar vector values which can make it less precise.

To create word embeddings you need a corpus and embedding method. The corpus gives a word a context which is what we use to deduce its meaning. The embedding method creates the embedding from the corpus. Many methods exist, however, here we will use modern machine learning models who are self-supervised as their training data (the corpus) is not labeled but does contain the context allowing us to extract the labels. When training word vectors there are some parameters who can be tuned such as the word vector's dimension.

#### Word embedding methods
A lot of word embedding methods exist. New methods are created to capture more and more meaning.

Here is a little history of word-embedding methods.

Google created in 2013 'word2vec' which uses a shallow neural network to learn word embeddings. It proposes two model architechtures. Continuous bag-of-words (CBOW) is a simple and efficient model that learns to predict a missing word given the surrounding words. Continuous skip gram or skip gram with negative sampling (SGNS) learns to predict a word surrounding a given input word.<br>
Stanford created in 2014 'Global Vectors' (GloVe) which uses a count matrix as we have seen before.<br>
Facebook created in 2016 'fastText'. It is based on the continuous skip gram model and takes in account the structure of words by representing words as n-grams of characters. This enables the model to support out of vocabulary words.

Some more advanced word embedding methods use deep neural network architectures to refine the representation of the words' meaning according to their context.<br>
For example Google created in 2018 BERT, Allen institute for AI created in 2018 ELMo, OpenAI created in 2018 GPT-2. You can download pre-trained embeddings of those models. 

#### Continuous Bag-of-Words Model
The objective of the CBOW model is to predict a missing word from the surrounding words. If two words are often used around similar contextual words in various sentences their meaning tends to be related.

The model needs to learn from training examples. Take the corpus 'I am happy because I am learning'. We can create a training example of context half-size (C) of 2 and window size of 5 for example. 'I am' and 'because I' would be the surrounding context words, while 'happy' would be the center word, and together the context and center words form the window of size 5. The context half-size (C) can be tuned as a model hyperparameter. To find next training examples the window can slide forward one word and here would thus become 'am happy because I am' with 'because' as center word.<br>
The model should take the context words as input and output the predicted center word.

##### Data preparation
Tokenization means splitting into words. Before CBOW we will clean and tokenize the corpus. In the first NLP course we already talked about data preparation, cleaning and tokenization. However, here we will go in more details.<br>
Words of corpus should be case insensitive, meaning you can convert the corpus to all lowercase characters.<br>
Punctuations should be handled. They can all be replaced by a special word of the vocabulary like '.' or some can even be dropped. Multiple punctuations like '???' can be seen as one entity and also be replaced by that special word '.'.<br>
Handle numbers by dropping them if they don't have meaning. However, certain numbers have meanings, such as 3.14 meaning Pi or 42 being a school, those can be left as is. If a lot of unique numbers have similar meanings such as different area codes, those can be replaced with a special token \<NUMBER\>.<br>
You also need to handle special characters like '@#$*'. It is usually safe to drop them.<br>
Special words such as emojis or hashtags like #nlp can be treated like individual words.

The context and central words need to be transformed into a mathematical form that can be consumed by the CBOW. We use one-hot-vectors, as explained above, for central words. For context words we create one vector for the whole context by taking the average of all words' one-hot-vectors.

##### Architecture
The CBOW model architecture is based on a shallow neural-network with an input layer, single hidden layer and an output layer. The input layer takes the context words vector and the output layer predicts the center word vector. Because one-hot-vectors are used, the input and output layers are of the size of the vocabulary's size. The hidden layer is given the same size as word embedding's size which is given to the model as hyperparameter. In between layers, weights and biases can be found. The activation function ReLU is used between input and hidden layer and softmax between hidden and output layer.<br>
![Screenshot 2024-03-26 at 20 37 33](https://github.com/artainmo/machine-learning/assets/53705599/5d4f37ec-852e-47e0-b5d2-bb4b57fd68bf)<br>
![Screenshot 2024-03-26 at 20 56 27](https://github.com/artainmo/machine-learning/assets/53705599/e169568a-41da-4b7b-b699-c9e1075f108f)<br>
[Learn more about how neural networks work.](https://github.com/artainmo/machine-learning/tree/main/supervised-learning%20and%20neural-networks)

Feeding multiple examples to the neural-network simultaneously is known as batch processing. M defines the batch size and is a model hyperparameter. You can join the examples' input vectors to form a matrix of m columns and feed it to the neural-network. Then the neural-network will output a matrix that contains the predicted center word vectors of the associated input examples.

In CBOW model when predicting the center word we output a vector that has the vocabulary's size like a one-hot-vector would. Softmax is used to transform the values in that vector into 0-1 probabilities. Each value at a particular index in that output vector thus represents the chance of the associated word being the center word.

##### Cost function
The cost function measures the degree of error between the predicted output and true label. The goal of the learning proccess is to find the model parameters that minimizes the cost.

CBOW uses the cross-entropy loss function which is often used during classification because it punishes misclassifications with a higher cost. It is also called log loss function.

y refers to correct answer vector. ŷ refers to prediction/output-vector. To calculate the log loss we will create a vector that contains the logarithm of all ŷ values. We will multiply that vector log(ŷ) with the initial y vector. We will sum all values of the vector 'y * log(ŷ)' and multiple the result with -1 to get the final cost value.

To calculate the cost of multiple examples, as we would with batch processing, we will average the cost of each individual training example. Thus, we sum the cost of each vector in matrix and divide that by the batch size m.

##### Forward and backward propagation
Forward propagation consists of the input going through the neural-network layers until it becomes a prediction. It mathematically looks like:<br>
Z<sub>1</sub> = W<sub>1</sub>X + B<sub>1</sub><br>
H = ReLU(Z<sub>1</sub>)<br>
Z<sub>2</sub> = W<sub>2</sub>H + B<sub>2</sub><br>
Ŷ = softmax(Z<sub>2</sub>)<br>

Backpropagation calculates the partial derivatives of the cost with respect to weights and biases using gradient descend. Weights and biases are updated by being substracted with the associated partial derivatives times [alpha](https://github.com/artainmo/machine-learning/tree/main/supervised-learning%20and%20neural-networks#LEARNING-RATE) with the goal of minizing the cost.

##### Extracting word embedding vectors
Word embeddings are vectors that carry the meaning of words based on the contextual words in the corpus. Word embeddings are not the output of the CBOW model they are a by-product of it.

After training the neural network, three word embedding representations can be extracted.<br>
The first possibility is to consider each column of W<sub>1</sub> as the embedding vector of a word of the vocabulary. Recall that matrix W<sub>1</sub> has V number of columns, so it has one column for each word in the vocabulary.<br>
The second possibility is to use each row of W<sub>2</sub> as the embedding vector of a word in the vocabulary. W<sub>2</sub> has V rows and thus one row for each vocabulary word.<br>
The last option is to use the average of W<sub>1</sub> columns and W<sub>2</sub> rows. The order of those rows/columns conincides with the order of the associated words in the vocabulary. Thus first row/column represents the first vocabulary word and so forth.

#### Evaluating word embeddings
Intrinsic and extrinsic evaluation are two types of evaluations, one is best used over another depending on the task you are trying to optimize for.

Intrinsic evaluation methods assess how well word embeddings capture the semantic (meaning) or syntactic (grammar) relationships between words.<br>
Those word relationships can be assesed via analogies. An example of a semantic analogy is 'France is to Paris as Italy is to \<?\>'. An example of a syntactic analogy is 'seen is to saw as been is to \<?\>'.<br>
A clustering algorithm that groups similar word embedding vectors can also be used to perform intrinsic evaluation.<br>
Visualizing the word embedding vectors in a graph can also be considered a simple intrinsic evaluation method as it allows you to see if similar words are close to each other.

To evaluate word embeddings with extrinsic evaluation you use the word embeddings to perform an external task which usually is the real-world task you initially needed the word embeddings for. Such an external task can be speech recognition for example, more specifically tasks like named entity recognition, parts-of-speech tagging, etc. The performance metric of this task will be used as a proxy evaluating the word embeddings. Extrinsic evaluation is the ultimate evaluation of word embeddings being useful in practice. However this type of evaluation is more time-consuming and provides less indications on where the problem lies if the performance is poor. 

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Probabilistic Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
