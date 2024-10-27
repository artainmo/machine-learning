# To silence the TensorFlow warnings, you can use the following code before you import the TensorFlow library.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import layers
from tensorflow.keras import losses
import re
import string
import matplotlib.pyplot as plt

# Setting the random seed allows you to have control over the (pseudo)random numbers. 
# When you are working with neural networks this is a good idea, so you can get reproducible results
seed = 42
# Sets the global random seed for numpy.
np.random.seed(seed)
# Sets the global random seed for TensorFlow.
tf.random.set_seed(seed)

data_dir = './data/aclImdb'

# Below, you will use the function tf.keras.utils.text_dataset_from_directory, 
# that generates a tf.data.Dataset from text files in a directory 
# that contains subdirectories who indicate classes for label classification.
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text_dataset_from_directory

# Here you have two main directories: one for train and one for test data.
# You load files from each to create training and test datasets.

# Create the training set. Use 80% of the data and keep the remaining 20% for the validation.
raw_training_set = tf.keras.utils.text_dataset_from_directory(
    f'{data_dir}/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    validation_split=0.2,
    subset='training',
    seed=seed
)
# Create the validation set. Use 20% of the data that was not used for training.
raw_validation_set = tf.keras.utils.text_dataset_from_directory(
    f'{data_dir}/train',
    labels='inferred',
    label_mode='int',
    batch_size=32,
    validation_split=0.2,
    subset='validation',
    seed=seed
)
# Create the test set.
raw_test_set = tf.keras.utils.text_dataset_from_directory(
    f'{data_dir}/test',
    labels='inferred',
    label_mode='int',
    batch_size=32,
)

# As subdirectories are named neg and pos, those should be the class names
print(f"Label 0 corresponds to {raw_training_set.class_names[0]}")
print(f"Label 1 corresponds to {raw_training_set.class_names[1]}")
# Output
# Label 0 corresponds to neg
# Label 1 corresponds to pos

# Take one batch from the dataset and print out the first three datapoints in the batch
for text_batch, label_batch in raw_training_set.take(1):
    for i in range(3):
        print(f"Review:\n {text_batch.numpy()[i]}")
        print(f"Label: {label_batch.numpy()[i]}\n")

#We will convert our datas to a format the neural-network understands using tf.keras.layers.TextVectorization.
#Which is a layer that converts text to vectors that can be fed to neural-networks.
#It can also automatically clean the datas by passing it another function that standardizes the text.
#After the standardization, the layer tokenizes the text (splits into words) and vectorizes it (converts from words to numbers).
#The 'output_sequence_length' is set to 250, which means that the layer will pad shorter sequences or truncate longer sequences, so they will al have the same length. This is done so that all the inout vectors are the same length and can be nicely put together into matrices.

# Set the maximum number of words, this will determine the vocabulary size
max_features = 10000

# Define the custom standardization function
def custom_standardization(input_data):
    # Convert all text to lowercase
    lowercase = tf.strings.lower(input_data)
    # Remove HTML tags
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    # Remove punctuation
    replaced = tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
        ''
    )
    return replaced

# Create a layer that you can use to convert text to vectors
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=250)

# .adapt() is used to let the model build a vocabulary.
train_text = raw_training_set.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
# Print out the vocabulary size
print(f"Vocabulary size: {len(vectorize_layer.get_vocabulary())}")

# Define the final function that you will use to vectorize the text.
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1) # This adds another dimension to your data and is very commonly used when processing data to add an additional dimension to accomodate for the batches.
    return vectorize_layer(text), label

#Now you can apply the vectorization function to vectorize all three datasets.
train_ds = raw_training_set.map(vectorize_text)
val_ds = raw_validation_set.map(vectorize_text)
test_ds = raw_test_set.map(vectorize_text)

#.cache() keeps data in memory after it's loaded off disk.
#.prefetch() overlaps data preprocessing and model execution while training.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#A Sequential model is appropriate when layers follow each other in a sequence and there are no additional connections.

embedding_dim = 16
# Create the model by calling tf.keras.Sequential, where the layers are given in a list.
model_sequential = tf.keras.Sequential([
    layers.Embedding(max_features, embedding_dim),
    #returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
    layers.GlobalAveragePooling1D(),
    #A dense layer with single output node and sigmoid activation function to make final prediction
    layers.Dense(1, activation='sigmoid') ])
# Print out the summary of the model
model_sequential.summary()

model_sequential.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

#For more complex models you don't want to use a sequential model. But instead a functional model.
#The Keras functional API is a way to create models that are more flexible than the keras.Sequential API.
#The functional API can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.

# Define the inputs
inputs = tf.keras.Input(shape=(None,))

# Define the first layer
embedding = layers.Embedding(max_features, embedding_dim)
# Call the first layer with inputs as the parameter
x = embedding(inputs)

# Define the second layer
pooling = layers.GlobalAveragePooling1D()
# Call the first layer with the output of the previous layer as the parameter
x = pooling(x)

# Define and call in the same line. (Same thing used two lines of code above
# for other layers. You can use any option you prefer.)
outputs = layers.Dense(1, activation='sigmoid')(x)
#The two-line alternative to the one layer would be:
# dense = layers.Dense(1, activation='sigmoid')
# x = dense(x)

# Create the model
model_functional = tf.keras.Model(inputs=inputs, outputs=outputs)
# Print out the summary of the model
model_functional.summary()

model_functional.compile(loss=losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

# Select which model you want to use and train. the results should be the same
model = model_functional 
# model = model_sequential

# Train the model

epochs = 25
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2
)

# Evaluate the model on the test dataset

loss, accuracy = model.evaluate(test_ds)

#When you trained the model, you saved the history in the history variable.
#You can use it to see how the training progressed.
def plot_metrics(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric.title())
    plt.legend([metric, f'val_{metric}'])
    plt.show()

plot_metrics(history, "accuracy")
plot_metrics(history, "loss")

#Expand previous model to create new model that makes predictions.
#Previously, you applied the TextVectorization layer to the dataset before feeding it to the model. 
#To simplify deploying the model, you can include the TextVectorization layer inside your model and then predict on raw strings.

# Make a new sequential model using the vectorization layer and the model you just trained.
export_model = tf.keras.Sequential([
  vectorize_layer,
  model]
)

# Compile the model
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

#Now you can use this model to predict on some of your own examples.
examples = ['this movie was very, very good', 'quite ok', 'the movie was not bad', 'bad', 'negative disappointed bad scary', 'this movie was stupid']
results = export_model.predict(examples, verbose=False)
for result, example in zip(results, examples):
    print(f'Result: {result[0]:.3f},   Label: {int(np.round(result[0]))},   Review: {example}')
