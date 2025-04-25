from collections import Counter

# build the vocabulary from M most frequent words
M = 3
word_counts = {'happy': 5, 'because': 3, 'i': 2, 'am': 2, 'learning': 3, '.': 1}
vocabulary = Counter(word_counts).most_common(M)
# remove the frequencies and leave just the words
vocabulary = [w[0] for w in vocabulary]
print(f"the new vocabulary containing {M} most frequent words: {vocabulary}\n")

# test if words in the input sentence are in the vocabulary
sentence = ['am', 'i', 'learning']
output_sentence = []
print(f"input sentence: {sentence}")
for w in sentence:
    # test if word w is in vocabulary
    if w in vocabulary:
        output_sentence.append(w)
    else:
        output_sentence.append('<UNK>')
print(f"output sentence: {output_sentence}")

# iterate through all word counts and print words with given frequency f
f = 3
word_counts = {'happy': 5, 'because': 3, 'i': 2, 'am': 2, 'learning':3, '.': 1}
for word, freq in word_counts.items():
    if freq == f:
        print(word)

# Add-k smoothing was described as a method for smoothing of the probabilities for previously unseen n-grams.
# In the code output bellow you'll see that a phrase that is in the training set gets the same probability as an unknown phrase.
# Thus add-k-smoothing is not the right method for this model.
def add_k_smoothing_probability(k, vocabulary_size, n_gram_count, n_gram_prefix_count):
    numerator = n_gram_count + k
    denominator = n_gram_prefix_count + k * vocabulary_size
    return numerator / denominator

trigram_probabilities = {('i', 'am', 'happy') : 2}
bigram_probabilities = {( 'i', 'am') : 10}
vocabulary_size = 5
k = 1

probability_known_trigram = add_k_smoothing_probability(k, vocabulary_size, trigram_probabilities[('i', 'am', 'happy')],
                           bigram_probabilities[( 'i', 'am')])
probability_unknown_trigram = add_k_smoothing_probability(k, vocabulary_size, 0, 0)
print(f"probability_known_trigram: {probability_known_trigram}")
print(f"probability_unknown_trigram: {probability_unknown_trigram}")

# Back-off is a model generalization method that leverages information from lower order n-grams in case information about the high order n-grams is missing. For example, if the probability of a trigram is missing, use bigram information and so on.
# pre-calculated probabilities of all types of n-grams
trigram_probabilities = {('i', 'am', 'happy'): 0}
bigram_probabilities = {( 'am', 'happy'): 0.3}
unigram_probabilities = {'happy': 0.4}
# this is the input trigram we need to estimate
trigram = ('are', 'you', 'happy')
# find the last bigram and unigram of the input
bigram = trigram[1: 3]
unigram = trigram[2]
print(f"besides the trigram {trigram} we also use bigram {bigram} and unigram ({unigram})\n")
# 0.4 is used as an example, experimentally found for web-scale corpuses when using the "stupid" back-off
lambda_factor = 0.4
probability_hat_trigram = 0
# search for first non-zero probability starting with trigram
# to generalize this for any order of n-gram hierarchy you could loop through the probability dictionaries instead of if/else cascade
if trigram not in trigram_probabilities or trigram_probabilities[trigram] == 0:
    print(f"probability for trigram {trigram} not found")
    if bigram not in bigram_probabilities or bigram_probabilities[bigram] == 0:
        print(f"probability for bigram {bigram} not found")
        if unigram in unigram_probabilities:
            print(f"probability for unigram {unigram} found\n")
            probability_hat_trigram = lambda_factor * lambda_factor * unigram_probabilities[unigram]
        else:
            probability_hat_trigram = 0
    else:
        probability_hat_trigram = lambda_factor * bigram_probabilities[bigram]
else:
    probability_hat_trigram = trigram_probabilities[trigram]
print(f"probability for trigram {trigram} estimated as {probability_hat_trigram}")

#The other method for using probabilities of lower order n-grams is the interpolation. In this case, you use weighted probabilities of n-grams of all orders every time, not just when high order information is missing.
# pre-calculated probabilities of all types of n-grams
trigram_probabilities = {('i', 'am', 'happy'): 0.15}
bigram_probabilities = {( 'am', 'happy'): 0.3}
unigram_probabilities = {'happy': 0.4}
# the weights come from optimization on a validation set
lambda_1 = 0.8
lambda_2 = 0.15
lambda_3 = 0.05
# this is the input trigram we need to estimate
trigram = ('i', 'am', 'happy')
# find the last bigram and unigram of the input
bigram = trigram[1: 3]
unigram = trigram[2]
print(f"besides the trigram {trigram} we also use bigram {bigram} and unigram ({unigram})\n")
# in the production code, you would need to check if the probability n-gram dictionary contains the n-gram
probability_hat_trigram = lambda_1 * trigram_probabilities[trigram] 
+ lambda_2 * bigram_probabilities[bigram]
+ lambda_3 * unigram_probabilities[unigram]
print(f"estimated probability of the input trigram {trigram} is {probability_hat_trigram}")

