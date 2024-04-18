#As you've learned, attention allows a seq2seq decoder to use information from each encoder step instead of just the final encoder hidden state. In the attention operation, the encoder outputs are weighted based on the decoder hidden state, then combined into one context vector. This vector is then used as input to the decoder to predict the next output step.

# Import the libraries and define the functions you will need for this lab
import numpy as np

def softmax(x, axis=0):
    """ Calculate softmax function for an array x along specified axis
    
        axis=0 calculates softmax across rows which means each column sums to 1 
        axis=1 calculates softmax across columns which means each row sums to 1
    """
    return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=axis), axis)

#The first step is to calculate the alignment scores. This is a measure of similarity between the decoder hidden state and each encoder hidden state.

#In practice, this is implemented as a feedforward neural network with two layers, where 𝑚 is the size of the layers in the alignment network.

hidden_size = 16
attention_size = 10
input_length = 5

np.random.seed(42)

# Synthetic vectors used to test
encoder_states = np.random.randn(input_length, hidden_size)
decoder_state = np.random.randn(1, hidden_size)

# Weights for the neural network, these are typically learned through training
# Use these in the alignment function below as the layer weights
layer_1 = np.random.randn(2 * hidden_size, attention_size)
layer_2 = np.random.randn(attention_size, 1)

# Implement this function. Replace None with your code. Solution at the bottom of the notebook
def alignment(encoder_states, decoder_state):
    # First, concatenate the encoder states and the decoder state
    single_decoder_state_for_each_encoder_state = np.repeat(decoder_state, 5, axis=0) #Transform decoder shape of (1, 16) to shape of (5, 16) to have similar shape as encoder
    inputs = np.concatenate((encoder_states, single_decoder_state_for_each_encoder_state), axis=1)
    assert inputs.shape == (input_length, 2 * hidden_size) #verify if correct shapes
    # Matrix multiplication of the concatenated inputs and layer_1, with tanh activation
    activations = np.tanh(np.matmul(inputs, layer_1))
    assert activations.shape == (input_length, attention_size)
    # Matrix multiplication of the activations with layer_2. Remember that you don't need tanh here
    scores = np.matmul(activations, layer_2)
    assert scores.shape == (input_length, 1)
    return scores

# The next step is to calculate the weights from the alignment scores. These weights should be between 0 and 1. You can use the softmax function (which is already implemented above) to get these weights from the alignment scores.

# The weights tell you the importance of each input word with respect to the decoder state. In this step, you use the weights to modulate the magnitude of the encoder vectors. Words with little importance will be scaled down relative to important words. Multiply each encoder vector by its respective weight to get the alignment vectors, then sum up the weighted alignment vectors to get the context vector.

def attention(encoder_states, decoder_state):
    """ Example function that calculates attention, returns the context vector

        Arguments:
        encoder_vectors: NxM numpy array, where N is the number of vectors and M is the vector length
        decoder_vector: 1xM numpy array, M is the vector length, much be the same M as encoder_vectors
    """
    # First, calculate the alignment scores
    scores = alignment(encoder_states, decoder_state)
    # Then take the softmax of the alignment scores to get a weight distribution
    weights = softmax(scores)
    # Multiply each encoder state by its respective weight
    weighted_scores = encoder_states * weights
    # Sum up weighted alignment vectors to get the context vector and return it
    context = np.sum(weighted_scores, axis=0)
    return context

context_vector = attention(encoder_states, decoder_state)
print(context_vector)
