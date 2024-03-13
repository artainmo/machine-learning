#In this lecture notebook you will create a matrix using some tag information and then modify it using different approaches. 

import numpy as np
import pandas as pd

#For this notebook you will be using a toy example including only three tags (or states). In a real world application there are many more tags which can be found here:
#https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# Define tags for Adverb, Noun and To (the preposition), respectively
tags = ['RB', 'NN', 'TO']

#counts the number of times a particular tag happened next to another
transition_counts = {
    ('NN', 'NN'): 16241,
    ('RB', 'RB'): 2263,
    ('TO', 'TO'): 2,
    ('NN', 'TO'): 5256,
    ('RB', 'TO'): 855,
    ('TO', 'NN'): 734,
    ('NN', 'RB'): 2431,
    ('RB', 'NN'): 358,
    ('TO', 'RB'): 200
}

# Store the number of tags in the 'num_tags' variable
num_tags = len(tags)
# Initialize a 3X3 numpy array with zeros
transition_matrix = np.zeros((num_tags, num_tags))
# Print shape of the matrix
print(transition_matrix.shape)

#Before filling the matrix with the values of the transition_counts dictionary you should sort the tags so that their placement in the matrix is consistent
sorted_tags = sorted(tags)

# Loop rows
for i in range(num_tags):
    # Loop columns
    for j in range(num_tags):
        # Define tag pair
        tag_tuple = (sorted_tags[i], sorted_tags[j])
        # Get frequency from transition_counts dict and assign to (i, j) position in the matrix
        transition_matrix[i, j] = transition_counts.get(tag_tuple)

#Use pandas dataframe to print out a pretty version of the matrix
def print_matrix(matrix):
    print(pd.DataFrame(matrix, index=sorted_tags, columns=sorted_tags))
print_matrix(transition_matrix)

#Each row can be normalized by dividing each value by the sum of row
# Compute sum of row for each row
rows_sum = transition_matrix.sum(axis=1, keepdims=True)
# Normalize transition matrix
transition_matrix = transition_matrix / rows_sum
# Notice that the normalization that was carried out forces the sum of each row to be equal to 1. 

# For a final example you are asked to modify each value of the diagonal of the matrix so that they are equal to the log of the sum of the current row plus the current value. 
import math
# Loop values in the diagonal
for i in range(num_tags):
    t_matrix_for[i, i] =  t_matrix_for[i, i] + math.log(rows_sum[i])

# Print matrix
print_matrix(t_matrix_for)
