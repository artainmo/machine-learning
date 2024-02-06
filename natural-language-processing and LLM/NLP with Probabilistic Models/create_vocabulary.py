#Create a tiny vocabulary from a tiny corpus

import re # regular expression library; for tokenization of words
from collections import Counter # collections library; counter: dict subclass for counting hashable objects
import matplotlib.pyplot as plt # for data visualization

# the tiny corpus of text ! 
text = 'red pink pink blue blue yellow ORANGE BLUE BLUE PINK' # 🌈
print(text)
print('string length : ',len(text))

#Preprocessing
# convert all letters to lower case
text_lowercase = text.lower()
print(text_lowercase)
print('string length : ',len(text_lowercase))
# some regex to tokenize the string to words and return them in a list, basically split string into list of words
words = re.findall(r'\w+', text_lowercase)
print(words)
print('count : ',len(words))

# create vocab
vocab = set(words)
print(vocab)
print('count : ',len(vocab))
# create vocab including word count
counts_a = dict()
for w in words:
    counts_a[w] = counts_a.get(w,0)+1 #retrieves the value associated with the key w in the dictionary counts_a. If w is not found in counts_a, it returns 0 (the second argument is the default value)
print(counts_a)
print('count : ',len(counts_a))
# create vocab including word count using collections.Counter
counts_b = dict()
counts_b = Counter(words)
print(counts_b)
print('count : ',len(counts_b))

# plot sorted word counts
d = {'blue': counts_b['blue'], 'pink': counts_b['pink'], 'red': counts_b['red'], 'yellow': counts_b['yellow'], 'orange': counts_b['orange']}
plt.bar(range(len(d)), list(d.values()), align='center', color=d.keys())
_ = plt.xticks(range(len(d)), list(d.keys()))
