import numpy as np
import matplotlib.pyplot as plt

# Set a sufficiently large n
n = 10000

# Generate n samples from a standard normal distribution
Z = np.random.randn(n)

# Calculate the empirical CDF
sorted_Z = np.sort(Z)
empirical_cdf = np.arange(1, n + 1) / float(n)

# Plot the empirical CDF
plt.step(sorted_Z, empirical_cdf)
plt.title('Empirical CDF of Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Empirical CDF')
plt.xlim(-3, 3)
plt.show()


# Question 6) 1. 
# # Defining the matrix A and vectors b and c
# A = np.array([[0, 2, 4], 
#               [2, 4, 2], 
#               [3, 3, 1]])
# b = np.array([-2, -2, -4])
# c = np.array([1, 1, 1])

# # Check if A is invertible (determinant not equal to zero)
# if np.linalg.det(A) != 0:
#     # Computing the inverse of A
#     A_inv = np.linalg.inv(A)

#     # Computing A_inv * b and A * c
#     A_inv_b = np.dot(A_inv, b)
#     A_c = np.dot(A, c)

#     # Printing the results
#     print("Inverse of A (A^-1):\n", A_inv)
#     print("Product of A^-1 and b (A^-1 * b):\n", A_inv_b)
#     print("Product of A and c (A * c):\n", A_c)
# else:
#     print("Matrix A is not invertible.")
