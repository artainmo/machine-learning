#Some common preprocessing steps for the language models include:
# * lowercasing the text
# * remove special characters
# * split text to list of sentences
# * split sentence into list words

import nltk               # NLP toolkit
import re                 # Library for Regular expression operations

nltk.download('punkt')    # Download the Punkt sentence tokenize

# change the corpus to lowercase
corpus = "Learning% makes 'me' happy. I am happy be-cause I am learning! :)"
corpus = corpus.lower()
# note that word "learning" will now be the same regardless of its position in the sentence
print(corpus)

# remove special characters
corpus = "learning% makes 'me' happy. i am happy be-cause i am learning! :)"
corpus = re.sub(r"[^a-zA-Z0-9.?! ]+", "", corpus)
print(corpus)
# Often, the special characters such as double quotes '"' or dash '-' are removed, and the interpunction such as full stop '.' or question mark '?' are left in the corpus.
# Note that this process gets rid of the happy face made with punctuations :). Remember that for sentiment analysis, this emoticon was very important. However, we will not consider it here.

# split text by a delimiter to array
input_date="Sat May  9 07:33:35 CEST 2020"
# get the date parts in array
date_parts = input_date.split(" ")
print(f"date parts = {date_parts}")
#get the time parts in array
time_parts = date_parts[4].split(":")
print(f"time parts = {time_parts}")

# tokenize the sentence into an array of words
sentence = 'i am happy because i am learning.'
tokenized_sentence = nltk.word_tokenize(sentence)
print(f'{sentence} -> {tokenized_sentence}')

def sentence_to_trigram(tokenized_sentence):
    """
    Prints all trigrams in the gives tokenized sentence.

    Args:
        tokenized_sentence: The words list.

    Returns:
        No output
    """
    # note that the last position of i is 3rd to the end
    for i in range(len(tokenized_sentence) - 3 + 1):
        # the sliding window starts at position i and contains 3 words
        trigram = tokenized_sentence[i : i + 3]
        print(trigram)

tokenized_sentence = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
print(f'List all trigrams of sentence: {tokenized_sentence}\n')
sentence_to_trigram(tokenized_sentence)

# when working with trigrams, you need to prepend 2 <s> and append one </s>
n = 3
tokenized_sentence = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
tokenized_sentence = ["<s>"] * (n - 1) + tokenized_sentence + ["</s>"]
print(tokenized_sentence)
