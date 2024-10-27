#Hash tables are data structures that allow indexing data to make lookup tasks more efficient. 
#In this part, you will see the implementation of the simplest hash function.

import numpy as np                # library for array and matrix manipulation
import pprint                     # utilities for console printing
from utils_nb import plot_vectors # helper function to plot vectors
import matplotlib.pyplot as plt   # visualization library

pp = pprint.PrettyPrinter(indent=4) # Instantiate a pretty printer

#The hash function is just the remainder of the integer division between each element and the desired number of buckets.
def basic_hash_table(value_l, n_buckets):
    def hash_function(value, n_buckets):
        return int(value) % n_buckets
    hash_table = {i:[] for i in range(n_buckets)} # Initialize all the buckets in the hash table as empty lists
    for value in value_l:
        hash_value = hash_function(value,n_buckets) # Get the hash key for the given value
        hash_table[hash_value].append(value) # Add the element to the corresponding bucket
    return hash_table

value_l = [100, 10, 14, 17, 97] # Set of values to hash
hash_table_example = basic_hash_table(value_l, n_buckets=10)
pp.pprint(hash_table_example)

#Multiplanes hash functions are other types of hash functions. Multiplanes hash functions are based on the idea of numbering every single region that is formed by the intersection of n planes. In the following code, we show the most basic forms of the multiplanes principle. First, with a single plane.
P = np.array([[1, 1]]) # Define a single plane.
fig, ax1 = plt.subplots(figsize=(8, 8)) # Create a plot
plot_vectors([P], axes=[2, 2], ax=ax1) # Plot the plane P as a vector
# Plot  random points.
for i in range(0, 10):
        v1 = np.array(np.random.uniform(-2, 2, 2)) # Get a pair of random numbers between -2 and 2
        side_of_plane = np.sign(np.dot(P, v1.T))
        # Color the points depending on the sign of the result of np.dot(P, point.T)
        if side_of_plane == 1:
            ax1.plot([v1[0]], [v1[1]], 'bo') # Plot blue points
        else:
            ax1.plot([v1[0]], [v1[1]], 'ro') # Plot red points
plt.show()

#The first thing to note is that the vector that defines the plane does not mark the boundary between the two sides of the plane. It marks the direction in which you find the 'positive' side of the plane.
#If we want to plot the separation plane, we need to plot a line that is perpendicular to our vector P. We can get such a line using a  90degrees rotation matrix.
P = np.array([[1, 2]])  # Define a single plane. You may change the direction
# Get a new plane perpendicular to P. We use a rotation matrix
PT = np.dot([[0, 1], [-1, 0]], P.T).T
fig, ax1 = plt.subplots(figsize=(8, 8)) # Create a plot with custom size
plot_vectors([P], colors=['b'], axes=[2, 2], ax=ax1) # Plot the plane P as a vector
# Plot the plane P as a 2 vectors.
# We scale by 2 just to get the arrows outside the current box
plot_vectors([PT * 4, PT * -4], colors=['k', 'k'], axes=[4, 4], ax=ax1)
# Plot 20 random points.
for i in range(0, 20):
        v1 = np.array(np.random.uniform(-4, 4, 2)) # Get a pair of random numbers between -4 and 4
        side_of_plane = np.sign(np.dot(P, v1.T)) # Get the sign of the dot product with P
        # Color the points depending on the sign of the result of np.dot(P, point.T)
        if side_of_plane == 1:
            ax1.plot([v1[0]], [v1[1]], 'bo') # Plot a blue point
        else:
            ax1.plot([v1[0]], [v1[1]], 'ro') # Plot a red point
plt.show()

#In the following section, we are going to define a hash function with a list of three custom planes in 2D.
P1 = np.array([[1, 1]])   # First plane 2D
P2 = np.array([[-1, 1]])  # Second plane 2D
P3 = np.array([[-1, -1]]) # Third plane 2D
P_l = [P1, P2, P3]  # List of arrays. It is the multi plane
# Vector to search
v = np.array([[2, 2]])
#The next function creates a hash value based on a set of planes.
def hash_multi_plane(P_l, v):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_of_plane(P,v)
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
    return hash_value
print(hash_multi_plane(P_l, v))

#Now we will handle planes represented in one matrix
np.random.seed(0)
num_dimensions = 2 # is 300 in assignment
num_planes = 3 # is 10 in assignment
random_planes_matrix = np.random.normal(
                       size=(num_planes,
                             num_dimensions))
print(random_planes_matrix)
v = np.array([[2, 2]])

def side_of_plane_matrix(P, v):
    dotproduct = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dotproduct)#Get a boolean value telling if the value in the cell is positive or negative
    return sign_of_dot_product

sides_l = side_of_plane_matrix(random_planes_matrix, v)
print(sides_l)

def hash_multi_plane_matrix(P, v, num_planes):
    sides_matrix = side_of_plane_matrix(P, v) # Get the side of planes for P and v
    hash_value = 0
    for i in range(num_planes):
        sign = sides_matrix[i].item() # Get the value inside the matrix cell
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i # sum 2^i * hash_i
    return hash_value
hash_multi_plane_matrix(random_planes_matrix, v, num_planes)

#This showed you how to make one set of random planes. You will make multiple sets of random planes in order to make the approximate nearest neighbors more accurate.
