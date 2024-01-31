import numpy as np #numpy is one of the most used libraries in Python for arrays/vectors/matrices manipulation

alist = [1, 2, 3, 4, 5]   #define a python list. It looks like an np array
narray = np.array([1, 2, 3, 4]) #define a numpy array

#Note the difference between a Python list and a NumPy array.
print(alist)
print(narray)
print(type(alist))
print(type(narray))
#Note that the '+' operator on numpy arrays perform an element-wise addition, while the same operation on python lists results in a list concatenation.
print(narray + narray)
print(alist + alist)
#In numpy '*' performs an element-wise multiplication, while on python lists this operator concatenates the same list multiple times..
print(narray * 3)
print(alist * 3)

#In numpy matrices are best created using 'np.array'.
npmatrix1 = np.array([narray, narray, narray]) #Matrix initialized with numpy arrays
npmatrix2 = np.array([alist, alist, alist]) #Matrix initialized with lists
npmatrix3 = np.array([narray, [1, 1, 1, 1], narray]) #Matrix initialized with both types
print(npmatrix1)
print(npmatrix2)
print(npmatrix3)
#Make sure when defining a matrix that all the rows are of the same length.

#For each element in the matrix, multiply by 2 and add 1
result = okmatrix * 2 + 1 
print(result)
#Add two compatible matrices
result1 = okmatrix + okmatrix
print(result1)
#Subtract two compatible matrices. This is called the difference vector
result2 = okmatrix - okmatrix
print(result2)
#The product operator * when used on arrays or matrices indicates element-wise multiplications. Do not confuse it with the dot product.
result = okmatrix * okmatrix 
print(result)

#In linear algebra, the transpose of a matrix is an operator that flips a matrix over its diagonal
#T denotes the transpose operations with numpy matrices
matrix3x2 = np.array([[1, 2], [3, 4], [5, 6]]) # Define a 3x2 matrix
print('Original matrix 3 x 2')
print(matrix3x2)
print('Transposed matrix 2 x 3')
print(matrix3x2.T)
#The transpose operation does not affect 1D arrays such as [1, 2, 3, 4] however it does affect 1x4 matrices such as [[1, 2, 3, 4]]

#Calculating the norm of vector or even of a matrix is a general operation when dealing with data. Numpy has a set of functions for linear algebra in the subpackage linalg, including the norm function.
nparray1 = np.array([1, 2, 3, 4]) # Define an array
norm1 = np.linalg.norm(nparray1)
nparray2 = np.array([[1, 2], [3, 4]]) # Define a 2 x 2 matrix. Note the 2 level of square brackets
norm2 = np.linalg.norm(nparray2) 
print(norm1)
print(norm2)
#Note that without any other parameter, the norm function treats the matrix as being just an array of numbers. However, it is possible to get the norm by rows or by columns.
nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) #Define a 3 x 2 matrix.
normByCols = np.linalg.norm(nparray2, axis=0) #Get the norm for each column. Returns 2 elements.
normByRows = np.linalg.norm(nparray2, axis=1) #Get the norm for each row. Returns 3 elements.
print(normByCols)
print(normByRows)

#The dot product takes two vectors and returns a single number.
nparray1 = np.array([0, 1, 2, 3]) # Define an array
nparray2 = np.array([4, 5, 6, 7]) # Define an array
flavor1 = np.dot(nparray1, nparray2) # Recommended way
print(flavor1)
flavor2 = np.sum(nparray1 * nparray2) # Ok way
print(flavor2)
flavor3 = nparray1 @ nparray2         # Geeks way
print(flavor3)
# As you never should do:             # Noobs way
flavor4 = 0
for a, b in zip(nparray1, nparray2):
    flavor4 += a * b
print(flavor4)

#Another general operation performed on matrices is the sum by rows or columns.
nparray2 = np.array([[1, -1], [2, -2], [3, -3]]) #Define a 3 x 2 matrix.
sumByCols = np.sum(nparray2, axis=0) #Get the sum for each column. Returns 2 elements
sumByRows = np.sum(nparray2, axis=1) #Get the sum for each row. Returns 3 elements
print('Sum by columns: ')
print(sumByCols)
print('Sum by rows:')
print(sumByRows)

#As with the sums, one can get the mean by rows or columns using the axis parameter.
nparray2 = np.array([[1, -1], [2, -2], [3, -3]]) # Define a 3 x 2 matrix. Chosen to be a matrix with 0 mean
mean = np.mean(nparray2) # Get the mean for the whole matrix
meanByCols = np.mean(nparray2, axis=0) # Get the mean for each column. Returns 2 elements
meanByRows = np.mean(nparray2, axis=1) # get the mean for each row. Returns 3 elements
print('Matrix mean: ')
print(mean)
print('Mean by columns: ')
print(meanByCols)
print('Mean by rows:')
print(meanByRows)

#Centering the attributes of a data matrix is another essential preprocessing step. Centering a matrix means to remove the column mean to each element inside the column. The mean by columns of a centered matrix is always 0.
nparray2 = np.array([[1, 1], [2, 2], [3, 3]]) # Define a 3 x 2 matrix.
nparrayCentered = nparray2 - np.mean(nparray2, axis=0) # Remove the mean for each column
print('Original matrix')
print(nparray2)
print('Centered by columns matrix')
print(nparrayCentered)
print('New mean by column')
print(nparrayCentered.mean(axis=0))


