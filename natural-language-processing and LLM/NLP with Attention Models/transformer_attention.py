import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import textwrap
wrapper = textwrap.TextWrapper(width=70)

# display_tensor() prints out the shape and the actual tensor.
def display_tensor(t, name):
    """Display shape and tensor"""
    print(f'{name} shape: {t.shape}\n')
    print(f'{t}\n')

def dot_product_attention(q, k, v, mask, scale=True):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

    Arguments:
        q (tf.Tensor): query of shape (..., seq_len_q, depth)
        k (tf.Tensor): key of shape (..., seq_len_k, depth)
        v (tf.Tensor): value of shape (..., seq_len_v, depth_v)
        mask (tf.Tensor): mask with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.
        scale (boolean): if True, the result is a scaled dot-product attention. Defaults to True.

    Returns:
        attention_output (tf.Tensor): the result of the attention function
    """
    # Multiply q and k transposed.
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk with the square root of dk
    if scale:
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        matmul_qk = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        matmul_qk = matmul_qk + (1. - mask) * -1e9
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.keras.activations.softmax(matmul_qk)
    # Multiply the attention weights by v
    attention_output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return attention_output

def causal_dot_product_attention(q, k, v, scale=True):
    """ Masked dot product self attention.
    Args:
        q (numpy.ndarray): queries.
        k (numpy.ndarray): keys.
        v (numpy.ndarray): values.
    Returns:
        numpy.ndarray: masked dot product self attention tensor.
    """
    # the mask array must have the same shape as tf.matmul(query, key_transposed).
    # Size of the penultimate dimension of the query
    mask_size = q.shape[-2]
    # Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
    mask = tf.experimental.numpy.tril(tf.ones((mask_size, mask_size)))
    return dot_product_attention(q, k, v, mask, scale=scale)

result = causal_dot_product_attention(q, k, v)
display_tensor(result, 'result')
