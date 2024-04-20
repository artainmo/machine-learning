#  Here, you will build an English-to-Portuguese neural machine translation (NMT) model using Long Short-Term Memory (LSTM) networks with attention.
# Machine translation is an important task in natural language processing and could be useful not only for translating one language to another but also for word sense disambiguation (e.g. determining whether the word "bank" refers to the financial bank, or the land alongside a river).

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Setting this env variable prevents TF warnings from showing up

import numpy as np
import tensorflow as tf
from collections import Counter
from utils import (sentences, train_data, val_data, english_vectorizer, portuguese_vectorizer,
                   masked_loss, masked_acc, tokens_to_text)
import w1_unittest

# We pre-processed the text in utils.py. Here are the performed steps.
# * Reading the raw data from the text files
# * Cleaning the data (using lowercase, adding space around punctuation, trimming whitespaces, etc)
# * Splitting it into training and validation sets
# * Adding the start-of-sentence and end-of-sentence tokens to every sentence
# * Tokenizing the sentences
# * Creating a Tensorflow dataset out of the tokenized sentences

portuguese_sentences, english_sentences = sentences
print(f"English (to translate) sentence:\n\n{english_sentences[-5]}\n")
print(f"Portuguese (translation) sentence:\n\n{portuguese_sentences[-5]}")

# Notice that you imported an english_vectorizer and a portuguese_vectorizer from utils.py. These were created using tf.keras.layers.TextVectorization and they provide interesting features such as ways to visualize the vocabulary and convert text into tokenized ids and vice versa.
print(f"First 10 words of the english vocabulary:\n\n{english_vectorizer.get_vocabulary()[:10]}\n")
print(f"First 10 words of the portuguese vocabulary:\n\n{portuguese_vectorizer.get_vocabulary()[:10]}")
# Output:
# First 10 words of the english vocabulary:
# ['', '[UNK]', '[SOS]', '[EOS]', '.', 'tom', 'i', 'to', 'you', 'the']
# First 10 words of the portuguese vocabulary:
# ['', '[UNK]', '[SOS]', '[EOS]', '.', 'tom', 'que', 'o', 'nao', 'eu']

# Size of the vocabulary
vocab_size_por = portuguese_vectorizer.vocabulary_size()
vocab_size_eng = english_vectorizer.vocabulary_size()
print(f"Portuguese vocabulary is made up of {vocab_size_por} words")
print(f"English vocabulary is made up of {vocab_size_eng} words")
# Output:
# Portuguese vocabulary is made up of 12000 words
# English vocabulary is made up of 12000 words

# You can define tf.keras.layers.StringLookup objects that will help you map from words to ids and vice versa. Do this for the portuguese vocabulary since this will be useful later on when you decode the predictions from your model.
word_to_id = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]")
id_to_word = tf.keras.layers.StringLookup(
    vocabulary=portuguese_vectorizer.get_vocabulary(),
    mask_token="",
    oov_token="[UNK]",
    invert=True)

unk_id = word_to_id("[UNK]")
sos_id = word_to_id("[SOS]")
eos_id = word_to_id("[EOS]")
baunilha_id = word_to_id("baunilha")
print(f"The id for the [UNK] token is {unk_id}")
print(f"The id for the [SOS] token is {sos_id}")
print(f"The id for the [EOS] token is {eos_id}")
print(f"The id for baunilha (vanilla) is {baunilha_id}")
# Output:
# The id for the [UNK] token is 1
# The id for the [SOS] token is 2
# The id for the [EOS] token is 3
# The id for baunilha (vanilla) is 7079

# Finally take a look at how the data that is going to be fed to the neural network looks like. Both train_data and val_data are of type tf.data.Dataset and are already arranged in batches of 64 examples. To get the first batch out of a tf dataset you can use the take method. To get the first example out of the batch you can slice the tensor and use the numpy method for nicer printing.
for (to_translate, sr_translation), translation in train_data.take(1):
    print(f"Tokenized english sentence:\n{to_translate[0, :].numpy()}\n\n")
    print(f"Tokenized portuguese sentence (shifted to the right):\n{sr_translation[0, :].numpy()}\n\n")
    print(f"Tokenized portuguese sentence:\n{translation[0, :].numpy()}\n\n")
# Output:
# Tokenized english sentence:
# [2 210 9 146 123  38 9 1672 4 3 0 0 0 0]
# Tokenized portuguese sentence (shifted to the right, starting with [SOS]):
#[2 1085 7 128 11 389 37 2038 4 0 0 0 0 0 0]
# Tokenized portuguese sentence:
# [1085 7 128 11 389 37 2038 4 3 0 0 0 0 0 0]

# The padding value is 0 as you can see.
# to_translate and sr_translation (shifted to the right translation) are features, the former used as input in encoder and latter as input in decoder when performing 'teacher forcing'. translation is the target.

# NMT model with attention

# In this assignment, the vocabulary sizes for English and Portuguese are the same. Therefore, we use a single constant VOCAB_SIZE throughout the notebook. While in other settings, vocabulary sizes could differ, that is not the case in our assignment.
VOCAB_SIZE = 12000
# The number of units in the LSTM layers (the same number will be used for all LSTM layers)
UNITS = 256

# Encoder

# GRADED CLASS: Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        """Initializes an instance of this class

        Args:
            vocab_size (int): Size of the vocabulary
            units (int): Number of units in the LSTM layer
        """
        super(Encoder, self).__init__()
        ### START CODE HERE ###
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="sum",
            layer=tf.keras.layers.LSTM(
                units=units,
                return_sequences=True
            ),
        )
        ### END CODE HERE ###

    def call(self, context):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): The sentence to translate

        Returns:
            tf.Tensor: Encoded sentence to translate
        """

        ### START CODE HERE ###
        # Pass the context through the embedding layer
        x = self.embedding(context)
        # Pass the output of the embedding through the RNN
        x = self.rnn(x)
        ### END CODE HERE ###
        return x

# Test your code!
encoder = Encoder(VOCAB_SIZE, UNITS)
encoder_output = encoder(to_translate)
print(f'Tensor of sentences in english has shape: {to_translate.shape}\n')
print(f'Encoder output has shape: {encoder_output.shape}')
w1_unittest.test_encoder(Encoder)

# CrossAttention

# GRADED CLASS: CrossAttention
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        """Initializes an instance of this class

        Args:
            units (int): Number of units in the LSTM layer
        """
        super().__init__()
        ### START CODE HERE ###
        self.mha = (
            tf.keras.layers.MultiHeadAttention(
                key_dim=units,
                num_heads=1
            )
        )
        ### END CODE HERE ###
        self.layernorm = tf.keras.layers.LayerNormalization() # normalization is also performed for better stability of the network
        self.add = tf.keras.layers.Add() # to preserve original dimension when combining matrices of different sizes

    def call(self, context, target):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): Encoded sentence to translate
            target (tf.Tensor): The embedded shifted-to-the-right translation

        Returns:
            tf.Tensor: Cross attention between context and target
        """
        ### START CODE HERE ###
        # Call the MH attention by passing in the query and value
        # For this case the query should be the translation and the value the encoded sentence to translate
        # Hint: Check the call arguments of MultiHeadAttention in the docs
        attn_output = self.mha(
            query=target,
            value=context
        )
        ### END CODE HERE ###
        x = self.add([target, attn_output])
        x = self.layernorm(x)
        return x

# Test your code!
attention_layer = CrossAttention(UNITS)
# The context (encoder_output) is already embedded so you need to do this for sr_translation:
sr_translation_embed = tf.keras.layers.Embedding(VOCAB_SIZE, output_dim=UNITS, mask_zero=True)(sr_translation)
attention_result = attention_layer(encoder_output, sr_translation_embed)
print(f'Tensor of contexts has shape: {encoder_output.shape}')
print(f'Tensor of translations has shape: {sr_translation_embed.shape}')
print(f'Tensor of attention scores has shape: {attention_result.shape}')
w1_unittest.test_cross_attention(CrossAttention)

# Decoder

# GRADED CLASS: Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, units):
        """Initializes an instance of this class

        Args:
            vocab_size (int): Size of the vocabulary
            units (int): Number of units in the LSTM layer
        """
        super(Decoder, self).__init__()
        ### START CODE HERE ###
        # The embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=units,
            mask_zero=True
        )
        # The RNN before attention
        self.pre_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True,
            return_state=True
        )  
        # The attention layer
        self.attention = CrossAttention(units)
        # The RNN after attention
        self.post_attention_rnn = tf.keras.layers.LSTM(
            units=units,
            return_sequences=True
        )  
        # The dense layer with logsoftmax activation
        self.output_layer = tf.keras.layers.Dense(
            # Should have the same number of units as the size of the vocabulary
            # since you expect it to compute the logits for every possible word in the vocabulary.
            units=vocab_size,  
            activation=tf.nn.log_softmax
        )
        ### END CODE HERE ###

    def call(self, context, target, state=None, return_state=False):
        """Forward pass of this layer

        Args:
            context (tf.Tensor): Encoded sentence to translate
            target (tf.Tensor): The shifted-to-the-right translation
            state (list[tf.Tensor, tf.Tensor], optional): Hidden state of the pre-attention LSTM. Defaults to None.
            return_state (bool, optional): If set to true return the hidden states of the LSTM. Defaults to False.

        Returns:
            tf.Tensor: The log_softmax probabilities of predicting a particular token
        """
        ### START CODE HERE ###
        # Get the embedding of the input
        x = self.embedding(target)
        # Pass the embedded input into the pre attention LSTM
        # Hints:
        # - The LSTM you defined earlier should return the output alongside the state (made up of two tensors)
        # - Pass in the state to the LSTM (needed for inference)
        x, hidden_state, cell_state = self.pre_attention_rnn(x, initial_state=state)
        # Perform cross attention between the context and the output of the LSTM (in that order)
        x = self.attention(context, x)
        # Do a pass through the post attention LSTM
        x = self.post_attention_rnn(x)
        # Compute the logits
        logits = self.output_layer(x)
        ### END CODE HERE ###
        if return_state:
            return logits, [hidden_state, cell_state]
        return logits

# Test your code!
decoder = Decoder(VOCAB_SIZE, UNITS)
# Logits are the outputs of a neural network before the activation function is applied. They are the unnormalized probabilities of the item belonging to a certain class. Logits are often used in classification tasks, where the goal is to predict the class label of an input.
logits = decoder(encoder_output, sr_translation)
print(f'Tensor of contexts has shape: {encoder_output.shape}')
print(f'Tensor of right-shifted translations has shape: {sr_translation.shape}')
print(f'Tensor of logits has shape: {logits.shape}')
w1_unittest.test_decoder(Decoder, CrossAttention)

# Translater

# GRADED CLASS: Translator
class Translator(tf.keras.Model):
    def __init__(self, vocab_size, units):
        """Initializes an instance of this class

        Args:
            vocab_size (int): Size of the vocabulary
            units (int): Number of units in the LSTM layer
        """
        super().__init__()
        ### START CODE HERE ###
        # Define the encoder with the appropriate vocab_size and number of units
        self.encoder = Encoder(vocab_size, units)
        # Define the decoder with the appropriate vocab_size and number of units
        self.decoder = Decoder(vocab_size, units)
        ### END CODE HERE ###

    def call(self, inputs):
        """Forward pass of this layer

        Args:
            inputs (tuple(tf.Tensor, tf.Tensor)): Tuple containing the context (sentence to translate) and the target (shifted-to-the-right translation)

        Returns:
            tf.Tensor: The log_softmax probabilities of predicting a particular token
        """
        ### START CODE HERE ###
        # In this case inputs is a tuple consisting of the context and the target, unpack it into single variables
        context, target = inputs
        # Pass the context through the encoder
        encoded_context = self.encoder(context)
        # Compute the logits by passing the encoded context and the target to the decoder
        logits = self.decoder(encoded_context, target)
        ### END CODE HERE ###
        return logits

# Do a quick check of your implementation
translator = Translator(VOCAB_SIZE, UNITS)
# Compute the logits for every word in the vocabulary
logits = translator((to_translate, sr_translation))
print(f'Tensor of sentences to translate has shape: {to_translate.shape}')
print(f'Tensor of right-shifted translations has shape: {sr_translation.shape}')
print(f'Tensor of logits has shape: {logits.shape}')
w1_unittest.test_translator(Translator, Encoder, Decoder)

# Training

def compile_and_train(model, epochs=20, steps_per_epoch=500):
    model.compile(optimizer="adam", loss=masked_loss, metrics=[masked_acc, masked_loss])
    history = model.fit(
        train_data.repeat(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=50,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)],
    )
    return model, history

# Train the translator (this takes some minutes so feel free to take a break)
trained_translator, history = compile_and_train(translator)

# Predictions

def generate_next_token(decoder, context, next_token, done, state, temperature=0.0):
    """Generates the next token in the sequence

    Args:
        decoder (Decoder): The decoder
        context (tf.Tensor): Encoded sentence to translate
        next_token (tf.Tensor): The predicted next token
        done (bool): True if the translation is complete
        state (list[tf.Tensor, tf.Tensor]): Hidden states of the pre-attention LSTM layer
        temperature (float, optional): The temperature that controls the randomness of the predicted tokens. Defaults to 0.0.

    Returns:
        tuple(tf.Tensor, np.float, list[tf.Tensor, tf.Tensor], bool): The next token, log prob of said token, hidden state of LSTM and if translation is done
    """
    # Get the logits and state from the decoder
    logits, state = decoder(context, next_token, state=state, return_state=True)
    # Trim the intermediate dimension 
    logits = logits[:, -1, :]
    # If temp is 0 then next_token is the argmax of logits
    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    # If temp is not 0 then next_token is sampled out of logits
    else:
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)
    # tf.squeeze - Given a tensor input, this operation returns a tensor of the same type with all dimensions of size 1 removed.
    logits = tf.squeeze(logits) 
    next_token = tf.squeeze(next_token)
    # Get the logit of the selected next_token
    logit = logits[next_token].numpy()
    # Reshape to (1,1) since this is the expected shape for text encoded as TF tensors
    next_token = tf.reshape(next_token, shape=(1,1))
    # If next_token is End-of-Sentence token you are done
    if next_token == eos_id:
        done = True
    return next_token, logit, state, done

# A sentence you wish to translate
eng_sentence = "I love languages"
# Convert it to a tensor
texts = tf.convert_to_tensor(eng_sentence)[tf.newaxis]
# Vectorize it and pass it through the encoder
context = english_vectorizer(texts).to_tensor()
context = encoder(context)
# SET STATE OF THE DECODER
# Next token is Start-of-Sentence since you are starting fresh
next_token = tf.fill((1,1), sos_id)
# Hidden and Cell states of the LSTM can be mocked using uniform samples
state = [tf.random.uniform((1, UNITS)), tf.random.uniform((1, UNITS))]
# You are not done until next token is EOS token
done = False
# Generate next token
next_token, logit, state, done = generate_next_token(decoder, context, next_token, done, state, temperature=0.5)
print(f"Next token: {next_token}\nLogit: {logit:.4f}\nDone? {done}")

# Translate

# GRADED FUNCTION: translate
def translate(model, text, max_length=50, temperature=0.0):
    """Translate a given sentence from English to Portuguese

    Args:
        model (tf.keras.Model): The trained translator
        text (string): The sentence to translate
        max_length (int, optional): The maximum length of the translation. Defaults to 50.
        temperature (float, optional): The temperature that controls the randomness of the predicted tokens. Defaults to 0.0.

    Returns:
        tuple(str, np.float, tf.Tensor): The translation, logit that predicted <EOS> token and the tokenized translation
    """
    # Lists to save tokens and logits
    tokens, logits = [], []
    ### START CODE HERE ###
    # PROCESS THE SENTENCE TO TRANSLATE
    # Convert the original string into a tensor
    text = tf.convert_to_tensor(text)[tf.newaxis] #tf.newaxis is similar to expand_dims() which adds a new axis.
    # Vectorize the text using the correct vectorizer
    context = english_vectorizer(text).to_tensor()
    # Get the encoded context (pass the context through the encoder)
    # Hint: Remember you can get the encoder by using model.encoder
    context = model.encoder(context)
    # INITIAL STATE OF THE DECODER
    # First token should be SOS token with shape (1,1)
    next_token = tf.fill((1, 1), sos_id)
    # Initial hidden and cell states should be tensors of zeros with shape (1, UNITS)
    state = [tf.zeros((1, UNITS)), tf.zeros((1, UNITS))]
    # You are done when you draw a EOS token as next token (initial state is False)
    done = False
    # Iterate for max_length iterations
    for _ in range(max_length):
        # Generate the next token
        try:
            next_token, logit, state, done = generate_next_token(
                decoder=model.decoder,
                context=context,
                next_token=next_token,
                done=done,
                state=state,
                temperature=temperature
            )
        except:
             raise Exception("Problem generating the next token")
        # If done then break out of the loop
        if done:
            break
        # Add next_token to the list of tokens
        tokens.append(next_token)
        # Add logit to the list of logits
        logits.append(logit)
    ### END CODE HERE ###
    # Concatenate all tokens into a tensor
    tokens = tf.concat(tokens, axis=-1)
    # Convert the translated tokens into text
    translation = tf.squeeze(tokens_to_text(tokens, id_to_word))
    translation = translation.numpy().decode()
    return translation, logits[-1], tokens

# Test
temp = 0.0
original_sentence = "I love languages"
translation, logit, tokens = translate(trained_translator, original_sentence, temperature=temp)
print(f"Temperature: {temp}\n\nOriginal sentence: {original_sentence}\nTranslation: {translation}\nTranslation tokens:{tokens}\nLogit: {logit:.3f}")
w1_unittest.test_translate(translate, trained_translator)

# Minimum Bayes-Risk Decoding

# This function will return any desired number of candidate translations alongside the log-probability for each one
def generate_samples(model, text, n_samples=4, temperature=0.6):
    samples, log_probs = [], []
    # Iterate for n_samples iterations
    for _ in range(n_samples):
        # Save the logit and the translated tensor
        _, logp, sample = translate(model, text, temperature=temperature)
        # Save the translated tensors
        samples.append(np.squeeze(sample.numpy()).tolist())
        # Save the logits
        log_probs.append(logp)
    return samples, log_probs

samples, log_probs = generate_samples(trained_translator, 'I love languages')
for s, l in zip(samples, log_probs):
    print(f"Translated tensor: {s} has logit: {l:.3f}")

# Now we need to compare those candidate translations
# For that we can use the widely used metric 'Jaccard similarity'.
def jaccard_similarity(candidate, reference):
    # Convert the lists to sets to get the unique tokens
    candidate_set = set(candidate)
    reference_set = set(reference)
    # Get the set of tokens common to both candidate and reference
    common_tokens = candidate_set.intersection(reference_set)
    # Get the set of all tokens found in either candidate or reference
    all_tokens = candidate_set.union(reference_set)
    # Compute the percentage of overlap (divide the number of common tokens by the number of all tokens)
    overlap = len(common_tokens) / len(all_tokens)
    return overlap

# Jaccard similarity is good but a more commonly used metric in machine translation is the ROUGE score. For unigrams, this is called ROUGE-1

# GRADED FUNCTION: rouge1_similarity
def rouge1_similarity(candidate, reference):
    """Computes the ROUGE 1 score between two token lists

    Args:
        candidate (list[int]): Tokenized candidate translation
        reference (list[int]): Tokenized reference translation

    Returns:
        float: Overlap between the two token lists
    """
    ### START CODE HERE ###
    # Make a frequency table of the candidate and reference tokens
    # Hint: use the Counter class (already imported)
    candidate_word_counts = Counter(candidate) # Will create a dictionnary with each unique token as key and the number of times it appears as value
    reference_word_counts = Counter(reference)
    # Initialize overlap at 0
    overlap = 0
    # Iterate over the tokens in the candidate frequency table
    # Hint: Counter is a subclass of dict and you can get the keys
    #       out of a dict using the keys method like this: dict.keys()
    for token in candidate_word_counts.keys():
        # Get the count of the current token in the candidate frequency table
        # Hint: You can access the counts of a token as you would access values of a dictionary
        token_count_candidate = candidate_word_counts[token]
        # Get the count of the current token in the reference frequency table
        # Hint: You can access the counts of a token as you would access values of a dictionary
        token_count_reference = reference_word_counts[token]
        # Update the overlap by getting the minimum between the two token counts above
        overlap += min(token_count_candidate, token_count_reference)
    # Compute the precision
    # Hint: precision = overlap / (number of tokens in candidate list)
    precision = overlap / len(candidate)
    # Compute the recall
    # Hint: recall = overlap / (number of tokens in reference list)
    recall = overlap / len(reference)
    if precision + recall != 0:
        # Compute the Rouge1 Score
        # Hint: This is equivalent to the F1 score
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    ### END CODE HERE ###
    return 0 # If precision + recall = 0 then return 0

# Test
l1 = [1, 2, 3]
l2 = [1, 2, 3, 4]
r1s = rouge1_similarity(l1, l2)
print(f"rouge 1 similarity between lists: {l1} and {l2} is {r1s:.3f}")
w1_unittest.test_rouge1_similarity(rouge1_similarity)

# You will now build a function to generate the overall score for a particular sample. As mentioned in the lectures, you need to compare each sample with all other samples.

# GRADED FUNCTION: average_overlap
def average_overlap(samples, similarity_fn):
    """Computes the arithmetic mean of each candidate sentence in the samples

    Args:
        samples (list[list[int]]): Tokenized version of translated sentences
        similarity_fn (Function): Similarity function used to compute the overlap

    Returns:
        dict[int, float]: A dictionary mapping the index of each translation to its score
    """
    # Initialize dictionary
    scores = {}
    # Iterate through all samples (enumerate helps keep track of indexes)
    for index_candidate, candidate in enumerate(samples):    
        ### START CODE HERE ###
        # Initially overlap is zero
        overlap = 0
        # Iterate through all samples (enumerate helps keep track of indexes)
        for index_sample, sample in enumerate(samples):
            # Skip if the candidate index is the same as the sample index
            if index_sample == index_candidate:
                continue
            # Get the overlap between candidate and sample using the similarity function
            sample_overlap = similarity_fn(candidate, sample)
            # Add the sample overlap to the total overlap
            overlap += sample_overlap
        ### END CODE HERE ###
        # Get the score for the candidate by computing the average
        score = overlap / (len(samples) - 1)
        # Only use 3 decimal points
        score = round(score, 3)
        # Save the score in the dictionary. use index as the key.
        scores[index_candidate] = score
    return scores

# Test with Jaccard similarity
l1 = [1, 2, 3]
l2 = [1, 2, 4]
l3 = [1, 2, 4, 5]
avg_ovlp = average_overlap([l1, l2, l3], jaccard_similarity)
print(f"average overlap between lists: {l1}, {l2} and {l3} using Jaccard similarity is:\n\n{avg_ovlp}")
# Output:
# average overlap between lists: [1, 2, 3], [1, 2, 4] and [1, 2, 4, 5] using Jaccard similarity is:
# {0: 0.45, 1: 0.625, 2: 0.575}
# Test with Rouge1 similarity
l1 = [1, 2, 3]
l2 = [1, 4]
l3 = [1, 2, 4, 5]
l4 = [5,6]
avg_ovlp = average_overlap([l1, l2, l3, l4], rouge1_similarity)
print(f"average overlap between lists: {l1}, {l2}, {l3} and {l4} using Rouge1 similarity is:\n\n{avg_ovlp}")
# Output:
# average overlap between lists: [1, 2, 3], [1, 4], [1, 2, 4, 5] and [5, 6] using Rouge1 similarity is:
# {0: 0.324, 1: 0.356, 2: 0.524, 3: 0.111}
w1_unittest.test_average_overlap(average_overlap)

# In practice, it is also common to see the weighted mean being used to calculate the overall score instead of just the arithmetic mean.
def weighted_avg_overlap(samples, log_probs, similarity_fn):
    # Scores dictionary
    scores = {}
    # Iterate over the samples
    for index_candidate, candidate in enumerate(samples):
        # Initialize overlap and weighted sum
        overlap, weight_sum = 0.0, 0.0
        # Iterate over all samples and log probabilities
        for index_sample, (sample, logp) in enumerate(zip(samples, log_probs)):
            # Skip if the candidate index is the same as the sample index
            if index_candidate == index_sample:
                continue
            # Convert log probability to linear scale
            sample_p = float(np.exp(logp))
            # Update the weighted sum
            weight_sum += sample_p
            # Get the unigram overlap between candidate and sample
            sample_overlap = similarity_fn(candidate, sample)
            # Update the overlap
            overlap += sample_p * sample_overlap
        # Compute the score for the candidate
        score = overlap / weight_sum
        # Only use 3 decimal points
        score = round(score, 3)
        # Save the score in the dictionary. use index as the key.
        scores[index_candidate] = score
    return scores

l1 = [1, 2, 3]
l2 = [1, 2, 4]
l3 = [1, 2, 4, 5]
log_probs = [0.4, 0.2, 0.5]
w_avg_ovlp = weighted_avg_overlap([l1, l2, l3], log_probs, jaccard_similarity)
print(f"weighted average overlap using Jaccard similarity is:\n\n{w_avg_ovlp}")
# Output:
# weighted average overlap using Jaccard similarity is:
# {0: 0.443, 1: 0.631, 2: 0.558}

# You will now put everything together in the the mbr_decode function below.

def mbr_decode(model, text, n_samples=5, temperature=0.6, similarity_fn=jaccard_similarity):
    # Generate samples
    samples, log_probs = generate_samples(model, text, n_samples=n_samples, temperature=temperature)
    # Compute the overlap scores
    scores = weighted_avg_overlap(samples, log_probs, similarity_fn)
    # Decode samples
    decoded_translations = [tokens_to_text(s, id_to_word).numpy().decode('utf-8') for s in samples]
    # Find the key with the highest score
    max_score_key = max(scores, key=lambda k: scores[k])
    # Get the translation
    translation = decoded_translations[max_score_key]
    return translation, decoded_translations

english_sentence = "I love languages"
translation, candidates = mbr_decode(trained_translator, english_sentence, n_samples=10, temperature=0.6)
print("Translation candidates:")
for c in candidates:
    print(c)
print(f"\nSelected translation: {translation}")
