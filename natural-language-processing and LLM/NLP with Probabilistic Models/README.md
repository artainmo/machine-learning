# NLP with Probabilistic Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Probabilistic Models](#DeepLearning.AI-Natural-Language-Processing-Specialization-NLP-with-Probabilistic-Models)
  - [Week 1: Autocorrect](Week-1-Autocorrect) 
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

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Probabilistic Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
