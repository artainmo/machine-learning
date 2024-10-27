import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf

# set random seeds to make this notebook easier to replicate
tf.keras.utils.set_random_seed(33)

import w2_unittest

# A few tags you might expect to see are:
#geo: geographical entity
#org: organization
#per: person
#gpe: geopolitical entity
#tim: time indicator
#art: artifact
#eve: event
#nat: natural phenomenon
#O: filler word (words who are not named entities)

def load_data(file_path):
    with open(file_path,'r') as file:
        data = np.array([line.strip() for line in file.readlines()])
    return data

train_sentences = load_data('data/large/train/sentences.txt')
train_labels = load_data('data/large/train/labels.txt')
val_sentences = load_data('data/large/val/sentences.txt')
val_labels = load_data('data/large/val/labels.txt')
test_sentences = load_data('data/large/test/sentences.txt')
test_labels = load_data('data/large/test/labels.txt')

# In this section, you will use tf.keras.layers.TextVectorization to transform the sentences into integers, so they can be fed into the model you will build later on.
# By default, standardize = 'lower_and_strip_punctuation', this means the parser will remove all punctuation and make everything lowercase. Note that this may influence the NER task, since an upper case in the middle of a sentence may indicate an entity. Furthermore, the sentences in the dataset are already split into tokens, and all tokens, including punctuation, are separated by a whitespace. The punctuations are also labeled. That said, you will use standardize = None so everything will just be split into single tokens and then mapped to a positive integer.
# Note that tf.keras.layers.TextVectorization will also pad the sentences. In this case, it will always pad using the largest sentence in the set you call it with.

# GRADED FUNCTION: get_sentence_vectorizer
def get_sentence_vectorizer(sentences):
    tf.keras.utils.set_random_seed(33) ## Do not change this line.
    """
    Create a TextVectorization layer for sentence tokenization and adapt it to the provided sentences.

    Parameters:
    sentences (list of str): Sentences for vocabulary adaptation.

    Returns:
    sentence_vectorizer (tf.keras.layers.TextVectorization): TextVectorization layer for sentence tokenization.
    vocab (list of str): Extracted vocabulary.
    """
    ### START CODE HERE ###
    # Define TextVectorization object with the appropriate standardize parameter
    sentence_vectorizer = tf.keras.layers.TextVectorization(standardize=None)
    # Adapt the sentence vectorization object to the given sentences
    sentence_vectorizer.adapt(sentences)
    # Get the vocabulary
    vocab = sentence_vectorizer.get_vocabulary()
    ### END CODE HERE ###
    return sentence_vectorizer, vocab

w2_unittest.test_get_sentence_vectorizer(get_sentence_vectorizer)

sentence_vectorizer, vocab = get_sentence_vectorizer(train_sentences)

# extract all the different tags in a given set of labels.
def get_tags(labels):
    tag_set = set() # Define an empty set
    for el in labels:
        for tag in el.split(" "):
            tag_set.add(tag)
    tag_list = list(tag_set)
    tag_list.sort()
    return tag_list

tags = get_tags(train_labels)

# generate a tag map, i.e., a mapping between the tags and positive integers.
def make_tag_map(tags):
    tag_map = {}
    for i,tag in enumerate(tags):
        tag_map[tag] = i
    return tag_map

# The tag_map is a dictionary that maps the tags that you could have to numbers. 
tag_map = make_tag_map(tags)

# In this section, you will pad the labels. TextVectorization already padded the sentences, so you must ensure that the labels are properly padded as well.
# You will pad the vectorized labels with the value -1. You will not use 0 to simplify loss masking and evaluation in further steps. This is because to properly classify one token, a log softmax transformation will be performed and the index with greater value will be the index label. Since index starts at 0, it is better to keep the label 0 as a valid index.

# GRADED FUNCTION: label_vectorizer
def label_vectorizer(labels, tag_map):
    """
    Convert list of label strings to padded label IDs using a tag mapping.

    Parameters:
    labels (list of str): List of label strings.
    tag_map (dict): Dictionary mapping tags to IDs.
    Returns:
    label_ids (numpy.ndarray): Padded array of label IDs.
    """
    label_ids = [] # It can't be a numpy array yet, since each sentence has a different size
    ### START CODE HERE ###
    # Each element in labels is a string of tags so for each of them:
    for element in labels:
        # Split it into single tokens. You may use .split function for strings. Be aware to split it by a blank space!
        tokens = element.split(" ")
        # Use the dictionaty tag_map to make the correspondence between tags and numbers.
        element_ids = []
        for token in tokens:
            element_ids.append(tag_map[token])
        # Append the found ids to corresponding to the current element to label_ids list
        label_ids.append(element_ids)
    # Pad the elements
    label_ids = tf.keras.utils.pad_sequences(sequences=label_ids, padding="post", value=-1)
    ### END CODE HERE ###
    return label_ids

w2_unittest.test_label_vectorizer(label_vectorizer)

# You will be using tf.data.Dataset class, which provides an optimized way to handle data to feed into a tensorflow model. It may be not as straightforward as a pandas dataset, but it avoids keeping all the data in memory, thus it makes the training faster.

def generate_dataset(sentences, labels, sentence_vectorizer, tag_map):
    sentences_ids = sentence_vectorizer(sentences)
    labels_ids = label_vectorizer(labels, tag_map = tag_map)
    dataset = tf.data.Dataset.from_tensor_slices((sentences_ids, labels_ids)) # converts any iterable into a Tensorflow dataset
    return dataset

train_dataset = generate_dataset(train_sentences,train_labels, sentence_vectorizer, tag_map)
val_dataset = generate_dataset(val_sentences,val_labels,  sentence_vectorizer, tag_map)
test_dataset = generate_dataset(test_sentences, test_labels,  sentence_vectorizer, tag_map)

# GRADED FUNCTION: NER
def NER(len_tags, vocab_size, embedding_dim = 50):
    """
    Create a Named Entity Recognition (NER) model.

    Parameters:
    len_tags (int): Number of NER tags (output classes).
    vocab_size (int): Vocabulary size.
    embedding_dim (int, optional): Dimension of embedding and LSTM layers (default is 50).

    Returns:
    model (Sequential): NER model.
    """
    ### START CODE HERE ### 
    model = tf.keras.Sequential(name = 'sequential') 
    # Add the tf.keras.layers.Embedding layer. Do not forget to mask out the zeros!
    model.add(tf.keras.layers.Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, mask_zero=True))
    # Add the LSTM layer. Make sure you are passing the right dimension (defined in the docstring above) 
    # and returning every output for the tf.keras.layers.LSTM layer and not the very last one.
    model.add(tf.keras.layers.LSTM(units=embedding_dim, return_sequences=True))
    # Add the final tf.keras.layers.Dense with the appropriate activation function. Remember you must pass the activation function itself ant not its call!
    # You must use tf.nn.log_softmax instead of tf.nn.log_softmax().
    model.add(tf.keras.layers.Dense(units=len_tags, activation=tf.nn.log_softmax))
    ### END CODE HERE ### 
    return model

w2_unittest.test_NER(NER)

# In some cases, however, there is one more step before getting the predicted labels. This may happen if, instead of passing the predicted labels, a vector of probabilities is passed. In such case, there is a need to perform an argmax for each prediction to find the appropriate predicted label. Such situations happen very often, therefore Tensorflow has a set of functions, with prefix Sparse, that performs this operation in the backend.

# GRADED FUNCTION: masked_loss
def masked_loss(y_true, y_pred):
    """
    Calculate the masked sparse categorical cross-entropy loss.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    loss (tensor): Calculated loss.
    """
    ### START CODE HERE ###
    # Calculate the loss for each item in the batch. Remember to pass the right arguments, as discussed above!
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=-1)
    # Use the previous defined function to compute the loss
    loss = loss_fn(y_true, y_pred)
    ### END CODE HERE ###
    return  loss

w2_unittest.test_masked_loss(masked_loss)

# Before training the model, you need to create your own function to compute the accuracy. Tensorflow has built-in accuracy metrics but you cannot pass values to be ignored. This will impact the calculations, since you must remove the padded values.
# remember to use only tensorflow operations. Even though numpy has every function you will need, to pass it as a loss function and/or metric function, you must use tensorflow operations, due to internal optimizations that Tensorflow performs for reliable fitting.

# GRADED FUNCTION: masked_accuracy
def masked_accuracy(y_true, y_pred):
    """
    Calculate masked accuracy for predicted labels.

    Parameters:
    y_true (tensor): True labels.
    y_pred (tensor): Predicted logits.

    Returns:
    accuracy (tensor): Masked accuracy.

    """
    ### START CODE HERE ###
    # Calculate the loss for each item in the batch.
    # You must always cast the tensors to the same type in order to use them in training. Since you will make divisions, it is safe to use tf.float32 data type.
    y_true = tf.cast(y_true, tf.float32) 
    # Create the mask, i.e., the values that will be ignored
    mask = 1.0 - np.equal(y_true, -1.0)
    mask = tf.cast(mask, tf.float32) 
    # Perform argmax to get the predicted values
    y_pred_class = tf.math.argmax(y_pred, axis=-1)
    y_pred_class = tf.cast(y_pred_class, tf.float32) 
    # Compare the true values with the predicted ones
    matches_true_pred  = tf.equal(y_true, y_pred_class)
    matches_true_pred = tf.cast(matches_true_pred , tf.float32) 
    # Multiply the acc tensor with the masks
    matches_true_pred *= mask
    # Compute masked accuracy (quotient between the total matches and the total valid values, i.e., the amount of non-masked values)
    masked_acc = tf.reduce_sum(matches_true_pred)/tf.reduce_sum(mask)
    ### END CODE HERE ### 
    return masked_acc

w2_unittest.test_masked_accuracy(masked_accuracy)

model = NER(len(tag_map), len(vocab))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss = masked_loss,
               metrics = [masked_accuracy])

tf.keras.utils.set_random_seed(33) ## Setting again a random seed to ensure reproducibility

BATCH_SIZE = 64
model.fit(train_dataset.batch(BATCH_SIZE),
          validation_data = val_dataset.batch(BATCH_SIZE),
          shuffle=True,
          epochs = 2)

# You will now evaluate on the test set.

# Convert the sentences into ids
test_sentences_id = sentence_vectorizer(test_sentences)
# Convert the labels into token ids
test_labels_id = label_vectorizer(test_labels,tag_map)
# Rename to prettify next function call
y_true = test_labels_id
y_pred = model.predict(test_sentences_id)
print(f"The model's accuracy in test set is: {masked_accuracy(y_true,y_pred).numpy():.4f}")

# In this section you will make a predictor function to predict the NER labels for any sentence.

# GRADED FUNCTION: predict
def predict(sentence, model, sentence_vectorizer, tag_map):
    """
    Predict NER labels for a given sentence using a trained model.

    Parameters:
    sentence (str): Input sentence.
    model (tf.keras.Model): Trained NER model.
    sentence_vectorizer (tf.keras.layers.TextVectorization): Sentence vectorization layer.
    tag_map (dict): Dictionary mapping tag IDs to labels.

    Returns:
    predictions (list): Predicted NER labels for the sentence.

    """
    ### START CODE HERE ### 
    # Convert the sentence into ids
    sentence_vectorized = sentence_vectorizer(sentence)
    # Expand its dimension to make it appropriate to pass to the model as the model expects batches
    sentence_vectorized = tf.expand_dims(sentence_vectorized, axis=0)
    # Get the model output
    output = model.predict(sentence_vectorized)
    # Get the predicted labels for each token, using argmax function and specifying the correct axis to perform the argmax
    outputs = np.argmax(output, axis =-1)
    # Next line is just to adjust outputs dimension. Since this function expects only one input to get a prediction, outputs will be something like [[1,2,3]]
    # so to avoid heavy notation below, let's transform it into [1,2,3]
    outputs = outputs[0] 
    # Get a list of all keys, remember that the tag_map was built in a way that each label id matches its index in a list
    labels = list(tag_map.keys()) 
    pred = [] 
    # Iterating over every predicted token in outputs list
    for tag_idx in outputs:
        pred_label = labels[tag_idx]
        pred.append(pred_label)
    ### END CODE HERE ### 
    return pred

w2_unittest.test_predict(predict, model, sentence_vectorizer, tag_map)

sentence = "Peter Parker , the White House director of trade and manufacturing policy of U.S , said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall , though he said it wouldn ’t necessarily come"
predictions = predict(sentence, model, sentence_vectorizer, tag_map)
for x,y in zip(sentence.split(' '), predictions):
    if y != 'O':
        print(x,y)


