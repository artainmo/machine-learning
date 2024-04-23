import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# There are two types of masks that are useful when building your Transformer network: the padding mask and the look-ahead mask. Both help the softmax computation give the appropriate weights to the words in your input sentence.

# Padding Mask

# When passing sequences into a transformer model, it is important that they are of uniform length. You can achieve this by padding the sequence with zeros, and truncating sentences that exceed the maximum length of your model.

# a boolean mask that specifies to which elements you must attend (1) and which elements you must ignore (0) and you do this by looking at all the zeros in the sequence. Then you use the mask to set the values of the vectors (corresponding to the zeros in the initial vector) close to negative infinity (-1e9).
# Imagine your input vector is [87, 600, 0, 0, 0]. This would give you a mask of [1, 1, 0, 0, 0]. When your vector passes through the attention mechanism, you get another (randomly looking) vector, let's say [1, 2, 3, 4, 5], which after masking becomes [1, 2, -1e9, -1e9, -1e9], so that when you take the softmax, the last three elements (where there were zeros in the input) don't affect the score.

def create_padding_mask(decoder_token_ids):
    """
    Creates a matrix mask for the padding cells
    
    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)
    
    Returns:
        mask (tf.Tensor): binary tensor of size (n, 1, m)
    """    
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits. 
    # this will allow for broadcasting later when comparing sequences
    return seq[:, tf.newaxis, :]

x = tf.constant([[7., 6., 0., 0., 0.], [1., 2., 3., 0., 0.], [3., 0., 0., 0., 0.]])
# Create the mask for x
mask = create_padding_mask(x)
# Extend the dimension of x to match the dimension of the mask
x_extended = x[:, tf.newaxis, :]
print("Softmax of non-masked vectors:\n")
print(tf.keras.activations.softmax(x_extended))
# Output:
# tf.Tensor([[[7.2959954e-01 2.6840466e-01 6.6530867e-04 6.6530867e-04 6.6530867e-04]]
# ....
print("\nSoftmax of masked vectors:\n")
# If you multiply (1 - mask) by -1e9 and add it to the sample input sequences, the zeros are essentially set to negative infinity.
print(tf.keras.activations.softmax(x_extended + (1 - mask) * -1.0e9))
# Output:
# tf.Tensor([[[0.7310586  0.26894143 0.         0.         0.        ]]
# ....

# Look-ahead Mask

# The look-ahead mask helps your model pretend that it correctly predicted a part of the output and see if, without looking ahead, it can correctly predict the next output.
# For example, if the expected correct output is [1, 2, 3] and you wanted to see if given that the model correctly predicted the first value it could predict the second value, you would mask out the second and third values. So you would input the masked sequence [1, -1e9, -1e9] and see if it could generate [1, 2, -1e9].

def create_look_ahead_mask(sequence_length):
    """
    Returns a lower triangular matrix filled with ones

    Arguments:
        sequence_length (int): matrix size

    Returns:
        mask (tf.Tensor): binary tensor of size (sequence_length, sequence_length)
    """
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask 

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)
# Output:
# [[[1., 0., 0.],
#   [1., 1., 0.],
#   [1., 1., 1.]]]
