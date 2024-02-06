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

Autocorrect is an application that changes misspelled words into the correct ones. It works first by identifying misspelled words, second find words who are a certain amount of edits away, lastly calculate word probabilities which calculates the chance of a word appearing in a certain context.

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

'Insert' has an edit cost of 1, 'delete' too and 'replace' has an edit cost of 2. Thus if 'play' needs to replace edits two times to become 'stay', the total edit cost will equal 2 + 2 = 4.

As strings become larger, it becomes harder to calculate the 'minimum edit distance'. This is why we will use the 'minimum edit distance' algorithm called 'dynamic programming'.

#### Dynamic programming
First we will create a distance matrix called D.

|   | # | s | t | a | y |
| - | - | - | - | - | - |
| # | 0 | 1 | 2 | 3 | 4 |
| p | 1 | 2 |   | 4 |   |
| l | 2 |   | 4 |   |   |
| a | 3 |   |   | 4 |   |
| y | 4 |   |   |   | 4 |

The # indicates an empty string. The minimum edit distance between two empty strings is 0. The minimum edit distance between p or s and an empty string is 1. The minimum edit distance between p and s is 2.<br>
For the transformation of pl into an empty string the minimum edit distance is 2, for the transformation of pla into an empty string the minimum edit distance is 3 and for the transformation of play into an empty string the minimum edit distance is 4.<br>

In D[2,2] we need to calculate the minimum edit distance between pl and st. For that we can take D[1,1] which is 2 and add to that the minimum edit distance between l and t which is also 2, thus D[2,2] equals 4. Similarly D[3,3] can be calculated by taking D[2,2] which equals 4 and adding to that the minimum edit distance between a and a which is 0, thus D[3,3] equals 4.<br>
If we would want to calculate for example D[1,3] we can take D[0,2] which equals 2 and add to that the difference minimum edit distance between p and a which is 2, thus D[1,3] equals 4.

D[m,n], in above example D[4,4] equaling 4, represents the minimum edit distance between the two compared words, in this example 'stay' and 'play'.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Probabilistic Models](https://www.coursera.org/learn/probabilistic-models-in-nlp)
