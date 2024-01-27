import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples, stopwords    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random
import re                                  # library for regular expression operations
import string                              # for string operations

# The sample dataset from NLTK is separated into positive and negative tweets. It contains 5000 positive tweets and 5000 negative tweets exactly. The exact match between these classes is not a coincidence. The intention is to have a balanced dataset.

#Download the dataset
nltk.download('twitter_samples')
#Select the set of positive and negative tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
#See dataset structure
print('Number of positive tweets: ', len(all_positive_tweets))
print('Number of negative tweets: ', len(all_negative_tweets))
print('The type of all_positive_tweets is: ', type(all_positive_tweets))
print('The type of a tweet entry is: ', type(all_negative_tweets[0]))
print('\033[92m' + all_positive_tweets[random.randint(0,5000)]) #See random tweet
print('\033[91m' + all_negative_tweets[random.randint(0,5000)] + '\033[0m')

#Download the stopwords from NLTK for preprocessing
nltk.download('stopwords')
#Import the english stop words list from NLTK
stopwords_english = stopwords.words('english')
#Our selected sample. Complex enough to exemplify each step
tweet = all_positive_tweets[2277]
print('\033[91m' + tweet)
#Remove old style retweet text "RT"
tweet2 = re.sub(r'^RT[\s]+', '', tweet)
#Remove hyperlinks
tweet2 = re.sub(r'https?://[^\s\n\r]+', '', tweet2)
#Remove hashtags, only removing the hash # sign from the word
tweet2 = re.sub(r'#', '', tweet2)
print(tweet2)
#Instantiate tokenizer class
tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
#Tokenize tweets
tweet_tokens = tokenizer.tokenize(tweet2)
print(tweet_tokens)
print('\033[0m', end='')
#The stop word list contains some words that could be important in some contexts. You might sometimes need to customize this list. For this exercise we will use the entire list.
print('Stop words:')
print(stopwords_english)
#Certain punctuations such as ':)' or '...' should be retained as they can indicate emotions. In other contexts, like medical analysis, these should also be removed.
print('Punctuation:')
print(string.punctuation)
#Remove stopwords and punctuations
tweets_clean = []
for word in tweet_tokens:
    if word not in stopwords_english and word not in string.punctuation:
        tweets_clean.append(word)
print("\033[91m", end='')
print(tweets_clean)
#In some cases, the stemming process produces words that are not correct spellings of the root word. For example, happi and sunni. That's because it chooses the most common stem for related words. For example, we can look at the set of words that comprises the different forms of happy: [happ]y, [happi]ness, [happi]er. We can see that the prefix happi is more commonly used. We cannot choose happ because it is the stem of unrelated words like happen.
#NLTK has different modules for stemming and we will be using the PorterStemmer module which uses the Porter Stemming Algorithm.
stemmer = nltk.stem.PorterStemmer()
tweets_stem = []
for word in tweets_clean:
    tweets_stem.append(stemmer.stem(word))
print("\033[92m", end='')
print(tweets_stem)


