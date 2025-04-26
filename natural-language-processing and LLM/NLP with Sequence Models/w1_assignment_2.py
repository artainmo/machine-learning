# In this lab, you'll delve into the world of text generation using Recurrent Neural Networks (RNNs). Your primary objective is to predict the next set of characters based on the preceding ones. This seemingly straightforward task holds immense practicality in applications like predictive text and creative writing.

import os
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import numpy as np
import random as  rnd

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input

from termcolor import colored

# set random seed
rnd.seed(32)

import w1_unittest

dirname = 'data/'
filename = 'shakespeare_data.txt'
lines = [] # storing all the lines in a variable.

counter = 0

with open(os.path.join(dirname, filename)) as files:
    for line in files:
        # remove leading and trailing whitespace
        pure_line = line.strip()#.lower()
        # if pure_line is not the empty string,
        if pure_line:
            # append it to the list
            lines.append(pure_line)

n_lines = len(lines)
print(f"Number of lines: {n_lines}")
# Output:
# Number of lines: 125097

# Identify and collect the unique characters that make up the text. This forms the basis of our vocabulary.
text = "\n".join(lines)
# The unique characters in the file
vocab = sorted(set(text))
vocab.insert(0,"[UNK]") # Add a special character for any unknown
vocab.insert(1,"") # Add the empty character for padding.

print(f'{len(vocab)} unique characters')
print(" ".join(vocab))
# Output:
# 82 unique characters

# GRADED FUNCTION: line_to_tensor
def line_to_tensor(line, vocab):
    """
    Converts a line of text into a tensor of integer values representing characters.

    Args:
        line (str): A single line of text.
        vocab (list): A list containing the vocabulary of unique characters.

    Returns:
        tf.Tensor(dtype=int64): A tensor containing integers (unicode values) corresponding to the characters in the `line`.
    """
    ### START CODE HERE ###
    # Split the input line into individual characters
    chars = tf.strings.unicode_split(line, input_encoding='UTF-8')
    # Map characters to their respective integer values using StringLookup
    ids = tf.keras.layers.StringLookup(vocabulary=vocab)(chars)
    ### END CODE HERE ###
    return ids

# UNIT TEST
w1_unittest.test_line_to_tensor(line_to_tensor)

def text_from_ids(ids, vocab):
    """
    Converts a tensor of integer values into human-readable text.

    Args:
        ids (tf.Tensor): A tensor containing integer values (unicode IDs).
        vocab (list): A list containing the vocabulary of unique characters.

    Returns:
        str: A string containing the characters in human-readable format.
    """
    # Initialize the StringLookup layer to map integer IDs back to characters
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)
    # Use the layer to decode the tensor of IDs into human-readable text
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

train_lines = lines[:-1000] # Leave the rest for training
eval_lines = lines[-1000:] # Create a holdout validation set

# Create a batch dataset from the input text

def split_input_target(sequence):
    """
    Splits the input sequence into two sequences, where one is shifted by one position.

    Args:
        sequence (tf.Tensor or list): A list of characters or a tensor.

    Returns:
        tf.Tensor, tf.Tensor: Two tensors representing the input and output sequences for the model.
    """
    # Create the input sequence by excluding the last character
    input_text = sequence[:-1]
    # Create the target sequence by excluding the first character
    target_text = sequence[1:]
    return input_text, target_text

# GRADED FUNCTION: create_batch_dataset
def create_batch_dataset(lines, vocab, seq_length=100, batch_size=64):
    """
    Creates a batch dataset from a list of text lines.

    Args:
        lines (list): A list of strings with the input data, one line per row.
        vocab (list): A list containing the vocabulary.
        seq_length (int): The desired length of each sample.
        batch_size (int): The batch size.

    Returns:
        tf.data.Dataset: A batch dataset generator.
    """
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000
    # For simplicity, just join all lines into a single line
    single_line_data  = "\n".join(lines)
    ### START CODE HERE ###
    # Convert your data into a tensor using the given vocab
    all_ids = line_to_tensor(single_line_data, vocab)
    # Create a TensorFlow dataset from the data tensor
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
    # Create a batch dataset
    data_generator = ids_dataset.batch(seq_length + 1, drop_remainder=True) 
    # Map each input sample using the split_input_target function
    dataset_xy = data_generator.map(split_input_target) 
    # Assemble the final dataset with shuffling, batching, and prefetching
    dataset = (                                   
        dataset_xy                                
        .shuffle(BUFFER_SIZE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)  
        )                        
    ### END CODE HERE ### 
    return dataset

# UNIT TEST
w1_unittest.test_create_batch_dataset(create_batch_dataset)

BATCH_SIZE = 64
dataset = create_batch_dataset(train_lines, vocab, seq_length=100, batch_size=BATCH_SIZE)
#This will produce pairs of input/output tensors each time the batch generator creates an entry.

# GRADED CLASS: GRULM
class GRULM(tf.keras.Model):
    """
    A GRU-based language model that maps from a tensor of tokens to activations over a vocabulary.

    Args:
        vocab_size (int, optional): Size of the vocabulary. Defaults to 256.
        embedding_dim (int, optional): Depth of embedding. Defaults to 256.
        rnn_units (int, optional): Number of units in the GRU cell. Defaults to 128.

    Returns:
        tf.keras.Model: A GRULM language model.
    """
    def __init__(self, vocab_size=256, embedding_dim=256, rnn_units=128):
        super().__init__(self)
        ### START CODE HERE ###
        # Create an embedding layer to map token indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # Define a GRU (Gated Recurrent Unit) layer for sequence modeling
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
        # Apply a dense layer with log-softmax activation to predict next tokens
        self.dense = tf.keras.layers.Dense(vocab_size, tf.nn.log_softmax)
        ### END CODE HERE ###

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        # Map input tokens to embedding vectors
        x = self.embedding(x, training=training)
        if states is None:
            # Get initial state from the GRU layer
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        # Predict the next tokens and apply log-softmax activation
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        else:
            return x

# Length of the vocabulary in StringLookup Layer
vocab_size = 82
# The embedding dimension
embedding_dim = 256
# RNN layers
rnn_units = 512
model = GRULM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units = rnn_units)
# testing your model
try:
    # Simulate inputs of length 100. This allows to compute the shape of all inputs and outputs of our network
    model.build(input_shape=(BATCH_SIZE, 100))
    model.call(Input(shape=(100)))
    model.summary()
except:
    print("\033[91mError! \033[0mA problem occurred while building your model. This error can occur due to wrong initialization of the return_sequences parameter\n\n")
    traceback.print_exc()

# the simplest way to choose the next character is by getting the index of the element with the highest likelihood
last_character = tf.math.argmax(example_batch_predictions[0][99])
print(last_character.numpy())

# Train

# GRADED FUNCTION: Compile model
def compile_model(model):
    """
    Sets the loss and optimizer for the given model

    Args:
        model (tf.keras.Model): The model to compile.

    Returns:
        tf.keras.Model: The compiled model.
    """
    ### START CODE HERE ###
    # Define the loss function. Use SparseCategoricalCrossentropy
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Define an Adam optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=0.00125)
    # Compile the model using the parametrized Adam optimizer and the SparseCategoricalCrossentropy funcion
    model.compile(optimizer=opt, loss=loss)
    ### END CODE HERE ###
    return model

EPOCHS = 10
# Compile the model
model = compile_model(model)
# Fit the model
history = model.fit(dataset, epochs=EPOCHS)

# # If you want, you can save the final model. Here is deactivated.
# output_dir = './your-model/'

# try:
#     shutil.rmtree(output_dir)
# except OSError as e:
#     pass

# model.save_weights(output_dir)

# Evaluation

# GRADED FUNCTION: log_perplexity
def log_perplexity(preds, target):
    """
    Function to calculate the log perplexity of a model.

    Args:
        preds (tf.Tensor): Predictions of a list of batches of tensors corresponding to lines of text.
        target (tf.Tensor): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: The log perplexity of the model.
    """
    PADDING_ID = 1
    ### START CODE HERE ###
    # Calculate log probabilities for predictions using one-hot encoding
    log_p = np.sum(preds * tf.one_hot(target, preds.shape[-1]), axis= -1)
    # Identify non-padding elements in the target
    non_pad = 1.0 - np.equal(target, PADDING_ID)          # You should check if the target equals to PADDING_ID
    # Apply non-padding mask to log probabilities to exclude padding
    log_p = log_p * non_pad                             # Get rid of the padding
    # Calculate the log perplexity by taking the sum of log probabilities and dividing it by the sum of non-padding elements
    log_ppx = np.sum(log_p, axis=1) / np.sum(non_pad, axis=1) # Remember to set the axis properly when summing up
    # Compute the mean of log perplexity
    log_ppx = np.mean(log_ppx) # Compute the mean of the previous expression
    ### END CODE HERE ###
    return -log_ppx

#UNIT TESTS
w1_unittest.test_test_model(log_perplexity)
#FAILS for unknown reason

#for line in eval_lines[1:3]:
eval_text = "\n".join(eval_lines)
eval_ids = line_to_tensor([eval_text], vocab)
input_ids, target_ids = split_input_target(tf.squeeze(eval_ids, axis=0))

preds, status = model(tf.expand_dims(input_ids, 0), training=False, states=None, return_state=True)

#Get the log perplexity
log_ppx = log_perplexity(preds, tf.expand_dims(target_ids, 0))
print(f'The log perplexity and perplexity of your model are {log_ppx} and {np.exp(log_ppx)} respectively')

# Generating Language with your Own Model

# Our model provides similar answers to same questions. 
# For it to generate different text sequences for similar inputs we can add an element of randomness into its predictions.
# The used technique is named 'random sampling'. The extent of randomness introduced into the predictions is regulated by a parameter called "temperature".

def temperature_random_sampling(log_probs, temperature=1.0):
    """Temperature Random sampling from a categorical distribution. The higher the temperature, the more 
       random the output. If temperature is close to 0, it means that the model will just return the index
       of the character with the highest input log_score
    
    Args:
        log_probs (tf.Tensor): The log scores for each character in the dictionary
        temperature (number): A value to weight the random noise. 
    Returns:
        int: The index of the selected character
    """
   # Generate uniform random numbers with a slight offset to avoid log(0)
    u = tf.random.uniform(minval=1e-6, maxval=1.0 - 1e-6, shape=log_probs.shape)
    # Apply the Gumbel distribution transformation for randomness
    g = -tf.math.log(-tf.math.log(u))
    # Adjust the logits with the temperature and choose the character with the highest score
    return tf.math.argmax(log_probs + g * temperature, axis=-1)

# UNGRADED CLASS: GenerativeModel
class GenerativeModel(tf.keras.Model):
    def __init__(self, model, vocab, temperature=1.0):
        """
        A generative model for text generation.

        Args:
            model (tf.keras.Model): The underlying model for text generation.
            vocab (list): A list containing the vocabulary of unique characters.
            temperature (float, optional): A value to control the randomness of text generation. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.vocab = vocab

    @tf.function
    def generate_one_step(self, inputs, states=None):
        """
        Generate a single character and update the model state.

        Args:
            inputs (string): The input string to start with.
            states (tf.Tensor): The state tensor.

        Returns:
            tf.Tensor, states: The predicted character and the current GRU state.
        """
        # Convert strings to token IDs.
        ### START CODE HERE ###
        # Transform the inputs into tensors
        input_ids = line_to_tensor(line, self.vocab)
        # Predict the sequence for the given input_ids. Use the states and return_state=True
        predicted_logits, states = self.model(input_ids, states, return_state=True)
        # Get only last element of the sequence
        predicted_logits = predicted_logits[0, -1, :]
        # Use the temperature_random_sampling to generate the next character.
        predicted_ids = temperature_random_sampling(predicted_logits, self.temperature)
        # Use the chars_from_ids to transform the code into the corresponding char
        predicted_chars = text_from_ids([predicted_ids], self.vocab)
        ### END CODE HERE ###
        # Return the characters and model state.
        return tf.expand_dims(predicted_chars, 0), states

    def generate_n_chars(self, num_chars, prefix):
        """
        Generate a text sequence of a specified length, starting with a given prefix.

        Args:
            num_chars (int): The length of the output sequence.
            prefix (string): The prefix of the sequence (also referred to as the seed).

        Returns:
            str: The generated text sequence.
        """
        states = None
        next_char = tf.constant([prefix])
        result = [next_char]
        for n in range(num_chars):
            next_char, states = self.generate_one_step(next_char, states=states)
            result.append(next_char)
        return tf.strings.join(result)[0].numpy().decode('utf-8')

w1_unittest.test_GenerativeModel(GenerativeModel, model, vocab)
# FAILS for unknown reason
