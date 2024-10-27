#In this notebook, you will apply linear algebra operations using numpy to find analogies between words manually.

import pandas as pd # Library for Dataframes
import numpy as np # Library for math functions
import pickle # Python object serialization library. Not secure

word_embeddings = pickle.load( open( "./data/word_embeddings_subset.p", "rb" ) )
print(len(word_embeddings)) # there should be 243 words that will be used in this assignment

#word_embeddings is a dictionary. Each word is the key to the entry, and the value is its corresponding vector presentation.
countryVector = word_embeddings['country'] #Get the vector representation for the word 'country'
print(type(countryVector)) #Print the type of the vector. Note it is a numpy array
print(countryVector) #Print the values of the vector.

#Get the vector for a given word:
def vec(w):
    return word_embeddings[w]

#These word embedding needs to be validated or at least understood because the performance of the derived model will strongly depend on its quality.
#we make a beautiful plot for the word embeddings of some words. Even if plotting the dots gives an idea of the words, the arrow representations help to visualize the vector's alignment as well.
import matplotlib.pyplot as plt # Import matplotlib
%matplotlib inline
words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
bag2d = np.array([vec(word) for word in words]) # Convert each word to its vector representation
fig, ax = plt.subplots(figsize = (10, 10)) # Create custom size image
col1 = 3 # Select the column for the x axis
col2 = 2 # Select the column for the y axis
# Print an arrow for each word
for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.005, head_length=0.005, fc='r', ec='r', width = 1e-5)
ax.scatter(bag2d[:, col1], bag2d[:, col2]); # Plot a dot for each word
# Add the word label over each dot in the scatter plot
for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))
plt.show()

#display the vector from 'village' to 'town' and the vector from 'sad' to 'happy'. Let us use numpy for these linear algebra operations.
words = ['sad', 'happy', 'town', 'village']
bag2d = np.array([vec(word) for word in words]) # Convert each word to its vector representation
fig, ax = plt.subplots(figsize = (10, 10)) # Create custom size image
col1 = 3 # Select the column for the x axe
col2 = 2 # Select the column for the y axe
# Print an arrow for each word
for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width=0.0005, head_length=0.0005, fc='r', ec='r', width = 1e-5)
# print the vector difference between village and town
village = vec('village')
town = vec('town')
diff = town - village
ax.arrow(village[col1], village[col2], diff[col1], diff[col2], fc='b', ec='b', width = 1e-5)
# print the vector difference between village and town
sad = vec('sad')
happy = vec('happy')
diff = happy - sad
ax.arrow(sad[col1], sad[col2], diff[col1], diff[col2], fc='b', ec='b', width = 1e-5)
ax.scatter(bag2d[:, col1], bag2d[:, col2]); # Plot a dot for each word
# Add the word label over each dot in the scatter plot
for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))
plt.show()

#Now, applying vector difference and addition, one can create a vector representation for a new word. For example, we can say that the vector difference between 'France' and 'Paris' represents the concept of Capital.
#One can move from the city of Madrid in the direction of the concept of Capital, and obtain something close to the corresponding country to which Madrid is the Capital.
capital = vec('France') - vec('Paris')
country = vec('Madrid') + capital
print(country[0:5]) # Print the first 5 values of the vector
#We can observe that the vector 'country' that we expected to be the same as the vector for Spain is not exactly it.
diff = country - vec('Spain')
print(diff[0:10])
#So, we have to look for the closest words in the embedding that matches the candidate country. If the word embedding works as expected, the most similar word must be 'Spain'.
keys = word_embeddings.keys()
data = []
for key in keys:
    data.append(word_embeddings[key])
embedding = pd.DataFrame(data=data, index=keys) # Create a dataframe out of the dictionary embedding. This facilitate the algebraic operations
def find_closest_word(v, k = 1):
    # Calculate the vector difference from each word to the input vector
    diff = embedding.values - v
    # Get the squared L2 norm of each difference vector.
    # It means the squared euclidean distance from each word to the input vector
    delta = np.sum(diff * diff, axis=1)
    # Find the index of the minimun distance in the array
    i = np.argmin(delta)
    # Return the row name for this item
    return embedding.iloc[i].name
find_closest_word(country)

#Predicting other countries
print(find_closest_word(vec('Italy') - vec('Rome') + vec('Madrid')))
print(find_closest_word(vec('Berlin') + capital))
print(find_closest_word(vec('Beijing') + capital))

#A whole sentence can be represented as a vector by summing all the word vectors that conform to the sentence.
doc = "Spain petroleum city king"
vdoc = [vec(x) for x in doc.split(" ")]
doc2vec = np.sum(vdoc, axis = 0)
print(doc2vec)

