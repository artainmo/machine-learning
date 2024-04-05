import numpy as np
from numpy import random
from time import perf_counter
import tensorflow as tf

def sigmoid(x): # Sigmoid function
    return 1.0 / (1.0 + np.exp(-x))

random.seed(10)                 # Random seed, so your results match ours
emb = 128                       # Embedding size
T = 256                         # Length of sequence
h_dim = 16                      # Hidden state dimension
h_0 = np.zeros((h_dim, 1))      # Initial hidden state

# Random initialization of weights (w1, w2, w3) and biases (b1, b2, b3)
w1 = random.standard_normal((h_dim, emb + h_dim))
w2 = random.standard_normal((h_dim, emb + h_dim))
w3 = random.standard_normal((h_dim, emb + h_dim))
b1 = random.standard_normal((h_dim, 1))
b2 = random.standard_normal((h_dim, 1))
b3 = random.standard_normal((h_dim, 1))
# Random initialization of input X
# Note that you add the third dimension (1) to achieve the batch representation.
X = random.standard_normal((T, emb, 1))

# Define the lists of weights as you will need them for the two different layers
weights_vanilla = [w1, b1]
weights_GRU = [w1.copy(), w2, w3, b1.copy(), b2, b3]

def forward_vanilla_RNN(inputs, weights): # Forward propagation for a a single vanilla RNN cell
    x, h_t = inputs
    # weights.
    wh, bh = weights
    # new hidden state
    h_t = np.dot(wh, np.concatenate([h_t, x])) + bh
    h_t = sigmoid(h_t)
    # We avoid implementation of y for clarity
    y = h_t
    # As you can see, we omitted the computation of 𝑦̂. This was done for the sake of simplicity, so you can focus on the way that hidden states are updated here and in the GRU cell.
    return y, h_t

def forward_GRU(inputs, weights): # Forward propagation for a single GRU cell
    x, h_t = inputs
    # weights.
    wu, wr, wc, bu, br, bc = weights
    # Update gate
    u = np.dot(wu, np.concatenate([h_t, x])) + bu
    u = sigmoid(u)
    # Relevance gate
    r = np.dot(wr, np.concatenate([h_t, x])) + br
    r = sigmoid(r)
    # Candidate hidden state
    c = np.dot(wc, np.concatenate([r * h_t, x])) + bc
    c = np.tanh(c)
    # New Hidden state h_t
    h_t = u * c + (1 - u) * h_t
    # We avoid implementation of y for clarity
    y = h_t
    return y, h_t

forward_GRU([X[1], h_0], weights_GRU)[0]

def scan(fn, elems, weights, h_0): # Abstract tensorflow function for RNN forward propagation
    h_t = h_0
    ys = []
    for x in elems:
        y, h_t = fn([x, h_t], weights)
        ys.append(y)
    return ys, h_t

ys, h_T = scan(forward_V_RNN, X, weights_vanilla, h_0)

# As you saw in the lectures, GRUs take more time to compute. This means that training and prediction would take more time for a GRU than for a vanilla RNN. However, GRUs allow you to propagate relevant information even for long sequences, so when selecting an architecture for NLP you should assess the tradeoff between computational time and performance.

# Create a GRU model in tensorflow

model_GRU = tf.keras.Sequential([
    tf.keras.layers.GRU(256, return_sequences=True, name='GRU_1_returns_seq'),
    tf.keras.layers.GRU(128, return_sequences=True, name='GRU_2_returns_seq'),
    tf.keras.layers.GRU(64, name='GRU_3_returns_last_only'),
    tf.keras.layers.Dense(10)
])

batch_size = 60
sequence_length = 50
word_vector_length = 40

input_data = tf.random.normal([batch_size, sequence_length, word_vector_length])

# Pass the data through the network
prediction = model_GRU(input_data)
# word_vector_length always needs to be the same for weight matrix multiplications
# batch_size and sequence_length can be of any size as they are not associated with weight matrix multiplications 

# Show the summary of the model
model_GRU.summary()

# Rather than passing data through the model, you can also specify the size of the data in an array and pass it to model.build(). This will build the model, taking into account the data shape. You can also pass None, where the data dimension may change.

model_GRU_2 = tf.keras.Sequential([
    tf.keras.layers.GRU(256, return_sequences=True, name='GRU_1_returns_seq'),
    tf.keras.layers.GRU(128, return_sequences=True, name='GRU_2_returns_seq'),
    tf.keras.layers.GRU(64, name='GRU_3_returns_last_only'),
    tf.keras.layers.Dense(10)
])

model_GRU_2.build([None, None, word_vector_length])

model_GRU_2.summary()
