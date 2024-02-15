import string
from collections import defaultdict

#A tagged dataset taken from the Wall Street Journal is provided in the file WSJ_02-21.pos.
with open("./data/WSJ_02-21.pos", 'r') as f:
    lines = f.readlines()

# Print columns for reference
print("\t\tWord", "\tTag\n")
# Print first five lines of the dataset
for i in range(5):
    print(f'line number {i+1}: {lines[i]}')

#List of POS tags: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

#Each line within the dataset has a word followed by its corresponding tag.
print(lines[0])
#Output: 'In\tIN\n'

#Now that you understand how the dataset is structured, you will create a vocabulary out of it.
#A vocabulary is made up of every word that appeared at least 2 times in the dataset.

# Get the words from each line in the dataset
words = [line.split('\t')[0] for line in lines]

#In case you aren't familiar with defaultdicts they are a special kind of dictionaries that return the "zero" value of a type if you try to access a key that does not exist. Since you want the frequencies of words, you should define the defaultdict with a type of int.

# Define defaultdict of type 'int'
freq = defaultdict(int)
# Count frequency of ocurrence for each word in the dataset
for word in words:
    freq[word] += 1

# Create the vocabulary by filtering the 'freq' dictionary
vocab = [k for k, v in freq.items() if (v > 1 and k != '\n')]

# Sort the vocabulary
vocab.sort()
# Print some random values of the vocabulary
for i in range(4000, 4005):
    print(vocab[i])

#At this point you will usually write the vocabulary into a file for future use, but that is out of the scope of this notebook.

def assign_unk(word):
    """
    Assign tokens to unknown words
    """

    # Punctuation characters
    # Try printing them out in a new cell!
    punct = set(string.punctuation)

    # Suffixes
    noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
    verb_suffix = ["ate", "ify", "ise", "ize"]
    adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
    adv_suffix = ["ward", "wards", "wise"]

    # Loop the characters in the word, check if any is a digit
    if any(char.isdigit() for char in word):
        return "--unk_digit--"

    # Loop the characters in the word, check if any is a punctuation character
    elif any(char in punct for char in word):
        return "--unk_punct--"

    # Loop the characters in the word, check if any is an upper case character
    elif any(char.isupper() for char in word):
        return "--unk_upper--"

    # Check if word ends with any noun suffix
    elif any(word.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"

    # Check if word ends with any verb suffix
    elif any(word.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"

    # Check if word ends with any adjective suffix
    elif any(word.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"

    # Check if word ends with any adverb suffix
    elif any(word.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"

    # If none of the previous criteria is met, return plain unknown
    return "--unk--"

#A POS tagger will always encounter words that are not within the vocabulary that is being used. By augmenting the dataset to include these unknown word tokens you are helping the tagger to have a better idea of the appropriate tag for these words.

def get_word_tag(line, vocab):
    # If line is empty return placeholders for word and tag
    if not line.split():
        word = "--n--"
        tag = "--s--"
    else:
        # Split line to separate word and tag
        word, tag = line.split()
        # Check if word is not in vocabulary
        if word not in vocab:
            # Handle unknown word
            tag = assign_unk(word)
    return word, tag


