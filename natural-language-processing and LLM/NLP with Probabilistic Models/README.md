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

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Probabilistic Models
](https://www.coursera.org/learn/probabilistic-models-in-nlp)
