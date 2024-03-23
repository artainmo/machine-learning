# NLP with Probabilistic Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Probabilistic Models](#DeepLearning.AI-Natural-Language-Processing-Specialization-NLP-with-Probabilistic-Models)
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
- [Resources](#Resources)

## DeepLearning.AI: Natural Language Processing Specialization: NLP with Probabilistic Models
### Week 1: Autocorrect
#### Introduction
In this course we will learn about word models and how to use them to predict word sequences. This allows autocorrection as well as web search suggestions. 

In the first week we will build an autocorrect system by using probabilities of character sequences. An important concept is 'minimum edit distance' which consists of evaluating the minimum amount of edits to change one word into another. 'Dynamic programming' is the 'minimum edit distance' algorithm we will use. It is an important programming concept which frequently comes up in interviews and could be used to solve a lot of optimization problems.

Autocorrect is an application that changes misspelled words into the correct ones. It works first by identifying misspelled words, second find words who are a certain amount of edits away, lastly calculate word probabilities which calculates the chance of a word appearing in a certain context.<br>
In this week's exercise we won't take the following in account, but autocorrect can also change words who exist but are used in wrong context. For example "Happy birthday deer friend", where 'deer' is an existing word but should be replaced with 'dear' in this context.

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
Word 'to' can only be found in O state. Transition probability from VB state to O state is 0.2 and emission probability from O state to word 'to' is 0.4. thus joint probability is 0.08.<br>
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

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Probabilistic Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
