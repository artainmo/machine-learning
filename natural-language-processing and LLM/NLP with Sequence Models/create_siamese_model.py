import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow import math
import numpy

# Setting random seeds
numpy.random.seed(10)

vocab_size = 500
model_dimension = 128

# Define the LSTM model
LSTM = Sequential()
LSTM.add(layers.Embedding(input_dim=vocab_size, output_dim=model_dimension))
LSTM.add(layers.LSTM(units=model_dimension, return_sequences = True))
LSTM.add(layers.AveragePooling1D()) # Takes mean across desired axis
LSTM.add(layers.Lambda(lambda x: math.l2_normalize(x))) # Lambda is used to apply a normalization function
# Instantiate keras tensors
input1 = layers.Input((None,))
input2 = layers.Input((None,))
# Concatenate two LSTMs together
conc = layers.Concatenate(axis=1)((LSTM(input1), LSTM(input2)))

# Use the Parallel combinator to create a Siamese model out of the LSTM
Siamese = Model(inputs=(input1, input2), outputs=conc)

# Print the summary of the model
Siamese.summary()


