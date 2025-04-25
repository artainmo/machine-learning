import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import os
import numpy as np
import pandas as pd
import random as rnd
import tensorflow as tf

# Set random seeds
rnd.seed(34)

import w3_unittest

# Importing the Data

data = pd.read_csv("questions.csv")
N = len(data)
print('Number of question pairs: ', N) #Output: 404351
data.head() #Display top values of pandas dataframe

N_train = 300000
N_test = 10240
data_train = data[:N_train]
data_test = data[N_train:N_train + N_test]
print("Train set:", len(data_train), "Test set:", len(data_test))
# Output: train set: 300000 Test set: 10240

# You need to build two sets of questions as input for the Siamese network, assuming that question 𝑞1𝑖(question 𝑖 in the first set) is a duplicate of 𝑞2𝑖(question 𝑖 in the second set), but all other questions in the second set are not duplicates of 𝑞1𝑖.
# Find the indexes with duplicate questions
td_index = data_train['is_duplicate'] == 1
td_index = [i for i, x in enumerate(td_index) if x]
print('Number of duplicate questions: ', len(td_index))
print('Indexes of first ten duplicate questions:', td_index[:10])
# Outputs:
# Number of duplicate questions:  111486
# Indexes of first ten duplicate questions: [5, 7, 11, 12, 13, 15, 16, 18, 20, 29]

# keep only the rows in the original training set that correspond to the rows where td_index is True
Q1_train = np.array(data_train['question1'][td_index])
Q2_train = np.array(data_train['question2'][td_index])

Q1_test = np.array(data_test['question1'])
Q2_test = np.array(data_test['question2'])
y_test  = np.array(data_test['is_duplicate'])

# Splitting the data to get validation set
cut_off = int(len(Q1_train) * 0.8)
train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
val_Q1, val_Q2 = Q1_train[cut_off:], Q2_train[cut_off:]

# Encode the inputs

tf.random.set_seed(0)
text_vectorization = tf.keras.layers.TextVectorization(output_mode='int',split='whitespace', standardize='strip_punctuation')
text_vectorization.adapt(np.concatenate((Q1_train,Q2_train))) # Create a vocabulary
print(f'Vocabulary size: {text_vectorization.vocabulary_size()}')
# Output: Vocabulary size: 36224

# As the vocabulary is known, tensors can be created for questions in integer representation
print('first question in the train set:\n')
print(Q1_train[0], '\n')
print('encoded version:')
print(text_vectorization(Q1_train[0]),'\n')
# Output:
# first question in the train set:
# Astrology: I am a Capricorn Sun Cap moon and cap rising...what does that say about me? 
# encoded version:
# tf.Tensor([6984 6 178 10 8988 2442 35393 761 13 6636 28205 31 28 483 45 98], shape=(16,), dtype=int64)

# The Siamese Network

# GRADED FUNCTION: Siamese
def Siamese(text_vectorizer, vocab_size=36224, d_feature=128):
    """Returns a Siamese model.

    Args:
        text_vectorizer (TextVectorization): TextVectorization instance, already adapted to your training data.
        vocab_size (int, optional): Length of the vocabulary. Defaults to 36224, which is the vocabulary size for your case.
        d_model (int, optional): Depth of the model. Defaults to 128.
        
    Returns:
        tf.model.Model: A Siamese model. 
    
    """
    ### START CODE HERE ###
    branch = tf.keras.models.Sequential(name='sequential') 
    # Add the text_vectorizer layer. This is the text_vectorizer you instantiated and trained before 
    branch.add(text_vectorizer)
    # Add the Embedding layer. Remember to call it 'embedding' using the parameter `name`
    branch.add(tf.keras.layers.Embedding(vocab_size, d_feature, name='embdedding'))
    # Add the LSTM layer, recall from W2 that you want the LSTM layer to return sequences, ot just one value. 
    # Remember to call it 'LSTM' using the parameter `name`
    branch.add(tf.keras.layers.LSTM(units=d_feature, name='LSTM', return_sequences=True))
    # Add the GlobalAveragePooling1D layer. Remember to call it 'mean' using the parameter `name`
    branch.add(tf.keras.layers.GlobalAveragePooling1D(name='mean'))
    # Add the normalizing layer using the Lambda function. Remember to call it 'out' using the parameter `name`
    branch.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x), name='out'))
    # Define both inputs. Remember to call then 'input_1' and 'input_2' using the `name` parameter. 
    # Be mindful of the data type and size
    input1 = tf.keras.layers.Input((1), name='input_1')
    input2 = tf.keras.layers.Input((1), name='input_2')
    # Define the output of each branch of your Siamese network. Remember that both branches have the same coefficients, 
    # but they each receive different inputs.
    branch1 = branch(input1)
    branch2 = branch(input2)
    # Define the Concatenate layer. You should concatenate columns, you can fix this using the `axis`parameter. 
    # This layer is applied over the outputs of each branch of the Siamese network
    conc = tf.keras.layers.Concatenate(axis=1, name='conc_1_2')([branch1, branch2]) 
    ### END CODE HERE ###
    return tf.keras.models.Model(inputs=[input1, input2], outputs=conc, name="SiameseModel")

# check your model
model = Siamese(text_vectorization, vocab_size=text_vectorization.vocabulary_size())
model.build(input_shape=None)
model.summary()
model.get_layer(name='sequential').summary()
# draw the model for a clearer view of the siamese network
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=True
)

# Test your function!
w3_unittest.test_Siamese(Siamese)

# Triplet Loss with Hard Negative Mining

# As this will be run inside Tensorflow, use all operation supplied by tf.math or tf.linalg, instead of numpy functions.
# However, we also put numpy functions in comments.
# GRADED FUNCTION: TripletLossFn
def TripletLossFn(v1, v2,  margin=0.25):
    """Custom Loss function.

    Args:
        v1 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q1.
        v2 (numpy.ndarray or Tensor): Array with dimension (batch_size, model_dimension) associated to Q2.
        margin (float, optional): Desired margin. Defaults to 0.25.

    Returns:
        triplet_loss (numpy.ndarray or Tensor)
    """
    ### START CODE HERE ###
    # use `tf.linalg.matmul` to take the dot product of the two batches.
    # Don't forget to transpose the second argument using `transpose_b=True`
    scores = tf.transpose(tf.linalg.matmul(v1, v2, transpose_b=True)) #Needs to be transposed for correct final output for unknown reason
    #scores2 = (np.dot(v1, v2.T)).T
    #scores2 = np.dot(v1, v2.T)
    # calculate new batch size and cast it as the same datatype as scores.
    batch_size = tf.cast(tf.shape(v1)[0], scores.dtype)
    #batch_size2 = len(scores2)
    # use `tf.linalg.diag_part` to grab the cosine similarity of all positive examples
    positive = tf.linalg.diag_part(scores)
    #positive2 = np.diagonal(scores2)
    # subtract the diagonal from scores. You can do this by creating a diagonal matrix with the values
    # of all positive examples using `tf.linalg.diag`
    negative_zero_on_duplicate = scores - tf.linalg.diag(positive)
    #negative_zero_on_duplicate2 = (1-np.eye(batch_size2)) * scores2
    # use `tf.math.reduce_sum` on `negative_zero_on_duplicate` for `axis=1` and divide it by `(batch_size - 1)`
    mean_negative = tf.math.reduce_sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    #mean_negative2 = np.sum(negative_zero_on_duplicate2, axis=1)/(batch_size2-1)
    # create a composition of two masks:
    # the first mask to extract the diagonal elements (make sure you use the variable batch_size here),
    # the second mask to extract elements in the negative_zero_on_duplicate matrix that are larger than the elements in the diagonal
    mask1 = tf.eye(batch_size) == 1 # Exclude diagonal containing positives
    mask2 = negative_zero_on_duplicate > tf.expand_dims(positive, 1) # Exclude similarity scores higher than the one in diagonal
    mask_exclude_positives = tf.cast(mask1|mask2, scores.dtype)
    #mask_exclude_positives2 = (np.identity(batch_size2) == 1)|(negative_zero_on_duplicate2 > positive2.reshape(batch_size2, 1))
    # multiply `mask_exclude_positives` with 2.0 and subtract it out of `negative_zero_on_duplicate`
    negative_without_positive = negative_zero_on_duplicate - 2.0*mask_exclude_positives
    #negative_without_positive2 = negative_zero_on_duplicate2 - (mask_exclude_positives2*2)
    # take the row by row `max` of `negative_without_positive`.
    # Hint: `tf.math.reduce_max(negative_without_positive, axis = None)`
    closest_negative = tf.math.reduce_max(negative_without_positive, axis=1)
    #closest_negative2 = negative_without_positive2.max(axis=1)
    # compute `tf.maximum` among 0.0 and `A`
    # A = subtract `positive` from `margin` and add `closest_negative`
    triplet_loss1 = tf.maximum(closest_negative - positive + margin, 0)
    #triplet_loss12 = np.maximum(0, margin-positive2+closest_negative2)
    # compute `tf.maximum` among 0.0 and `B`
    # B = subtract `positive` from `margin` and add `mean_negative`
    triplet_loss2 = tf.maximum(mean_negative - positive + margin, 0)
    #triplet_loss22 = np.maximum(0, margin-positive2+mean_negative2)
    # add the two losses together and take the `tf.math.reduce_sum` of it
    triplet_loss = tf.math.reduce_sum(triplet_loss1 + triplet_loss2)
    #triplet_loss_2 = np.sum(triplet_loss12 + triplet_loss22)
    ### END CODE HERE ###
    return triplet_loss
#For keras to recognize it as a loss function you need to take 'out' which coincides with the output of the siamese network
#which is the concatenation of each subnetwork's output.
def TripletLoss(labels, out, margin=0.25):
    _, out_size = out.shape # get embedding size
    v1 = out[:,:int(out_size/2)] # Extract v1 from out
    v2 = out[:,int(out_size/2):] # Extract v2 from out
    return TripletLossFn(v1, v2, margin=margin)

# Test your function!
w3_unittest.test_TripletLoss(TripletLoss)
# Passes two tests and fails one for unknown reason

# Train

train_dataset = tf.data.Dataset.from_tensor_slices(((train_Q1, train_Q2),tf.constant([1]*len(train_Q1))))
val_dataset = tf.data.Dataset.from_tensor_slices(((val_Q1, val_Q2),tf.constant([1]*len(val_Q1))))

# GRADED FUNCTION: train_model
def train_model(Siamese, TripletLoss, text_vectorizer, train_dataset, val_dataset, d_feature=128, lr=0.01, train_steps=5):
    """Training the Siamese Model

    Args:
        Siamese (function): Function that returns the Siamese model.
        TripletLoss (function): Function that defines the TripletLoss loss function.
        text_vectorizer: trained instance of `TextVecotrization`
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        d_feature (int, optional) = size of the encoding. Defaults to 128.
        lr (float, optional): learning rate for optimizer. Defaults to 0.01
        train_steps (int): number of epochs

    Returns:
        tf.keras.Model
    """
    ## START CODE HERE ###
    # Instantiate your Siamese model
    model = Siamese(text_vectorizer,
                    vocab_size=text_vectorization.vocabulary_size(),
                    d_feature=d_feature)
    # Compile the model
    model.compile(loss=TripletLoss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
    # Train the model
    model.fit(train_dataset,
              epochs=train_steps,
              validation_data=val_dataset)
    ### END CODE HERE ###
    return model

# Test your function!
w3_unittest.test_train_model(train_model, Siamese, TripletLoss)

# Evaluation

# Load a pretrained model for this exercise
model = tf.keras.models.load_model('model/trained_model.keras', safe_mode=False, compile=False)
# Show the model architecture
model.summary()

# GRADED FUNCTION: classify
def classify(test_Q1, test_Q2, y_test, threshold, model, batch_size=64, verbose=True):
    """Function to test the accuracy of the model.

    Args:
        test_Q1 (numpy.ndarray): Array of Q1 questions. Each element of the array would be a string.
        test_Q2 (numpy.ndarray): Array of Q2 questions. Each element of the array would be a string.
        y_test (numpy.ndarray): Array of actual target.
        threshold (float): Desired threshold
        model (tensorflow.Keras.Model): The Siamese model.
        batch_size (int, optional): Size of the batches. Defaults to 64.

    Returns:
        float: Accuracy of the model
        numpy.array: confusion matrix
    """
    y_pred = []
    test_gen = tf.data.Dataset.from_tensor_slices(((test_Q1, test_Q2),None)).batch(batch_size=batch_size)
    ### START CODE HERE ###
    pred = model.predict(test_gen)
    _, n_feat = pred.shape
    v1 = pred[:,:int(n_feat/2)]
    v2 = pred[:,int(n_feat/2):]
    # Compute the cosine similarity. Using `tf.math.reduce_sum`.
    # as l2 normalized, ||v1[j]||==||v2[j]||==1 so only dot product is needed
    # Don't forget to use the appropriate axis argument.
    d = tf.math.reduce_sum(v1*v2, axis=1) # takes the dot product between v1 and v2. Equivalent to np.dot(v1, v2)
    # Check if d>threshold to make predictions
    y_pred = tf.cast(d > threshold, tf.float64)
    # take the average of correct predictions to get the accuracy
    accuracy = tf.math.reduce_sum(tf.cast(y_test == y_pred, tf.int32)) / len(y_pred)
    # compute the confusion matrix using `tf.math.confusion_matrix`
    cm = tf.math.confusion_matrix(y_test, y_pred)
    ### END CODE HERE ###
    return accuracy, cm

# Test your function!
w3_unittest.test_classify(classify, model)

# Testing with your own questions

# GRADED FUNCTION: predict
def predict(question1, question2, threshold, model, verbose=False):
    """Function for predicting if two questions are duplicates.

    Args:
        question1 (str): First question.
        question2 (str): Second question.
        threshold (float): Desired threshold.
        model (tensorflow.keras.Model): The Siamese model.
        verbose (bool, optional): If the results should be printed out. Defaults to False.

    Returns:
        bool: True if the questions are duplicates, False otherwise.
    """
    generator = tf.data.Dataset.from_tensor_slices((([question1], [question2]),None)).batch(batch_size=1)
    ### START CODE HERE ###
    # Call the predict method of your model and save the output into v1v2
    v1v2 = model.predict(generator)
    # Extract v1 and v2 from the model output
    v1 = v1v2[:,:int(v1v2.shape[1]/2)]
    v2 = v1v2[:,int(v1v2.shape[1]/2):]
    # Take the dot product to compute cos similarity of each pair of entries, v1, v2
    # Since v1 and v2 are both vectors, use the function tf.math.reduce_sum instead of tf.linalg.matmul
    d = tf.math.reduce_sum(v1*v2, axis=1)
    # Is d greater than the threshold?
    res = d > threshold
    ### END CODE HERE ###
    if(verbose):
        print("Q1  = ", question1, "\nQ2  = ", question2)
        print("d   = ", d.numpy())
        print("res = ", res.numpy())
    return res.numpy()

# Test your function!
w3_unittest.test_predict(predict, model)
