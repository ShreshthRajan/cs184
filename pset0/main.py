import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Improve plot aesthetics with Seaborn
sns.set()

# Set a sufficiently large n (assumed to be 10000 here)
n = 10000

# Generate n samples from a standard normal distribution (mean 0, variance 1)
Z = np.random.randn(n)

# Calculate the empirical CDF of Z
sorted_Z = np.sort(Z)
empirical_cdf_Z = np.arange(1, n + 1) / float(n)

# Define k values
k_values = [1, 8, 64, 512]

# Plot the empirical CDF for each k
plt.figure(figsize=(10, 6))
plt.step(sorted_Z, empirical_cdf_Z, label='Standard Normal')

for k in k_values:
    Y_k = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1. / k), axis=1)
    sorted_Y_k = np.sort(Y_k)
    empirical_cdf_Y_k = np.arange(1, n + 1) / float(n)
    plt.step(sorted_Y_k, empirical_cdf_Y_k, label=f'Y({k})')

plt.title('Empirical CDFs of Standard Normal and Y(k)')
plt.xlabel('x')
plt.ylabel('Empirical CDF')
plt.legend()
plt.xlim(-3, 3)

# Save the plot as a PNG file in the current directory
plt.savefig("empirical_cdf_combined.png")

# Display the plot
plt.show()


# part A
# # Set a sufficiently large n (assumed to be 10000 here)
# # You should verify this n based on the condition from problem 3.5 if applicable
# n = 10000

# # Generate n samples from a standard normal distribution (mean 0, variance 1)
# # This simulates the random variables Z_i ~ N(0, 1) for i = 1, ..., n
# Z = np.random.randn(n)

# # Calculate the empirical CDF of the generated samples
# # The empirical CDF at a point x is the proportion of samples less than or equal to x
# sorted_Z = np.sort(Z)
# empirical_cdf = np.arange(1, n + 1) / float(n)

# # Plot the empirical CDF
# # The step plot represents the cumulative distribution function of the generated samples
# plt.step(sorted_Z, empirical_cdf)
# plt.title('Empirical CDF of Standard Normal Distribution')
# plt.xlabel('x')
# plt.ylabel('Empirical CDF')
# plt.xlim(-3, 3) # Limiting the x-axis to [-3, 3] to focus on the central part of the distribution

# # Save the plot as a PNG file in the current directory
# plt.savefig("empirical_cdf.png")

# # Display the plot
# plt.show()


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
