import numpy as np
import tensorflow as tf

def cosine_similarity(v1, v2):
    numerator = tf.math.reduce_sum(v1*v2) # takes the dot product between v1 and v2. Equivalent to np.dot(v1, v2)
    denominator = tf.math.sqrt(tf.math.reduce_sum(v1*v1) * tf.math.reduce_sum(v2*v2))
    return numerator / denominator

# Two batches of vectors example
v1_1 = np.array([1.0, 2.0, 3.0])
v1_2 = np.array([9.0, 8.0, 7.0])
v1_3 = np.array([-1.0, -4.0, -2.0])
v1_4 = np.array([1.0, -7.0, 2.0])
v1 = np.vstack([v1_1, v1_2, v1_3, v1_4])
# add some noise to create approximate duplicate
v2_1 = v1_1 + np.random.normal(0, 2, 3)  
v2_2 = v1_2 + np.random.normal(0, 2, 3)
v2_3 = v1_3 + np.random.normal(0, 2, 3)
v2_4 = v1_4 + np.random.normal(0, 2, 3)
v2 = np.vstack([v2_1, v2_2, v2_3, v2_4])

# batch size
b = len(v1)

# Similarity scores

# Option 1 : nested loops and the cosine similarity function
sim_1 = np.zeros([b, b])  # empty array to take similarity scores
for row in range(0, sim_1.shape[0]):
    for col in range(0, sim_1.shape[1]):
        sim_1[row, col] = cosine_similarity(v2[row], v1[col]).numpy()

# Now, you can repeat the procedure applying vectorization, so the computations are more efficient
# Option 2 : vector normalization and dot product
def norm(x):
    return tf.math.l2_normalize(x, axis=1) # use tensorflow built in normalization
sim_2 = tf.linalg.matmul(norm(v2), norm(v1), transpose_b=True)

# Calculate the mean negative and the closest negative

# numpy implementation

# Hardcoded matrix of similarity scores
sim_hardcoded = np.array(
    [
        [0.9, -0.8, 0.3, -0.5],
        [-0.4, 0.5, 0.1, -0.1],
        [0.3, 0.1, -0.4, -0.8],
        [-0.5, -0.2, -0.7, 0.5],
    ]
)

sim = sim_hardcoded

# Batch size
b = sim.shape[0]

# Positives
# All the s(A,P) values : similarities from duplicate question pairs (aka Positives)
# These are along the diagonal
sim_ap = np.diag(sim)

# Negatives
# all the s(A,N) values : similarities the non duplicate question pairs (aka Negatives)
# These are in the off diagonals
sim_an = sim - np.diag(sim_ap)

# Mean negative
# Average of the s(A,N) values for each row
mean_neg = np.sum(sim_an, axis=1, keepdims=True) / (b - 1)

# Closest negative
# Max s(A,N) that is <= s(A,P) for each row
mask_1 = np.identity(b) == 1            # mask to exclude the diagonal
mask_2 = sim_an > sim_ap.reshape(b, 1)  # mask to exclude sim_an > sim_ap
mask = mask_1 | mask_2
sim_an_masked = np.copy(sim_an)         # create a copy to preserve sim_an
sim_an_masked[mask] = -2
closest_neg = np.max(sim_an_masked, axis=1, keepdims=True)

# tensorflow implementation

sim = sim_hardcoded

# Positives
# All the s(A,P) values : similarities from duplicate question pairs (aka Positives)
# These are along the diagonal
sim_ap = tf.linalg.diag_part(sim) # this is just a 1D array of diagonal elements
# tf.linalg.diag makes a diagonal matrix given an array
print(tf.linalg.diag(sim_ap), "\n")

# Negatives
# all the s(A,N) values : similarities the non duplicate question pairs (aka Negatives)
# These are in the off diagonals
sim_an = sim - tf.linalg.diag(sim_ap)

# Mean negative
# Average of the s(A,N) values for each row
mean_neg = tf.math.reduce_sum(sim_an, axis=1) / (b - 1)

# Closest negative
# Max s(A,N) that is <= s(A,P) for each row
mask_1 = tf.eye(b) == 1            # mask to exclude the diagonal
mask_2 = sim_an > tf.expand_dims(sim_ap, 1)  # mask to exclude sim_an > sim_ap
mask = tf.cast(mask_1 | mask_2, tf.float64)
sim_an_masked = sim_an - 2.0*mask
closest_neg = tf.math.reduce_max(sim_an_masked, axis=1)

# Calculate the loss

# Alpha margin
alpha = 0.25

# Modified triplet loss
# Loss 1
l_1 = tf.maximum(mean_neg - sim_ap + alpha, 0)
# Loss 2
l_2 = tf.maximum(closest_neg - sim_ap + alpha, 0)
# Loss full
l_full = l_1 + l_2
# Cost
cost = tf.math.reduce_sum(l_full)
