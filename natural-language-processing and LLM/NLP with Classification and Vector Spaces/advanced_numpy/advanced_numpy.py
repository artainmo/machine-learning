#There are three main vector transformations: Scaling, Translation, Rotation
#The rotation operation changes the direction of a vector, leaving unaffected its dimensionality and its norm

import numpy as np                     # Import numpy for array manipulation
import matplotlib.pyplot as plt        # Import matplotlib for charts
from utils_nb import plot_vectors      # Function to plot vectors (arrows)

R = np.array([[-2, 0], # Create a 2 x 2 matrix
              [0, 2]])
x = np.array([[1, 1]]) # Create a row vector as a NumPy array with a single row
#The dot product between a square matrix and the transpose of a row vector produces a rotation and scaling of the original vector.
y = np.dot(R, x.T)
plot_vectors([x], axes=[4, 4], fname='transform_x.svg')
plot_vectors([x, y], axes=[4, 4], fname='transformx_and_y.svg')

#Rotation matrices are of the form:
#𝑅𝑜=[𝑐𝑜𝑠𝜃 −𝑠𝑖𝑛𝜃
#    𝑠𝑖𝑛𝜃 𝑐𝑜𝑠𝜃]
#In the next cell, we define a rotation matrix that rotates vectors counterclockwise by 100degrees.
angle = 100 * (np.pi / 180) # Convert degrees to radians
Ro = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])
x2 = np.array([[2, 2]])    # Row vector as a NumPy array
y2 = np.dot(Ro, x2.T)
print('Rotation matrix')
print(Ro)
print('\nRotated vector')
print(y2)
print('\n x2 norm', np.linalg.norm(x2))
print('\n y2 norm', np.linalg.norm(y2))
print('\n Rotation matrix norm', np.linalg.norm(Ro))
#The norm of the input vector is the same as the norm of the output vector. Rotation matrices do not modify the norm of the vector, only its direction.

#We will calculate the norm
A = np.array([[2, 2],
              [2, 2]])
A_squared = np.square(A)
A_norm = np.sqrt(np.sum(A_squared))
#That was the extended version of the np.linalg.norm() function. You can check that it yields the same result.
print('Frobenius norm of the Rotation matrix')
print(np.sqrt(np.sum(Ro * Ro)), '== ', np.linalg.norm(Ro))
