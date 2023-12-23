import numpy as np


# Define the cost function as per the given equation.
def cost_function(H, DER, P, a, b, c, e, theta, P_min):
    """
    Calculate the cost function F1 based on the provided parameters.

    :param H: int - the upper limit for the summation over t
    :param DER: int - the upper limit for the summation over k
    :param P: array - the power values for each t and k, should be a 2D array of shape (H, DER)
    :param a, b, c, e: arrays - the coefficient values for each k, should be 1D arrays of length DER
    :param theta: array - the theta values for each k, should be a 1D array of length DER
    :param P_min: array - the P_min values for each k, should be a 1D array of length DER
    :return: float - the computed cost
    """
    F1 = 0
    # Perform the double summation
    for t in range(H):
        for k in range(DER):
            F1 += (a[k] * (P[t, k] ** 2) + b[k] * P[t, k] + c[k]) + e[k] * np.sin(
                theta[k] * (P_min[k] - P[t, k])
            )
    return F1


# Example inputs to the function
# These are arbitrary values for the purpose of demonstrating the function implementation.
# In practice, these should be replaced with actual values provided by the user or another part of the program.
H = 5  # Replace with the actual value
DER = 3  # Replace with the actual value
P = np.array(
    [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]
)  # Replace with the actual value
a = np.array([0.1, 0.2, 0.3])  # Replace with the actual value
b = np.array([1, 1.5, 2])  # Replace with the actual value
c = np.array([0, 0, 0])  # Replace with the actual value
e = np.array([0.01, 0.02, 0.03])  # Replace with the actual value
theta = np.array([np.pi / 4, np.pi / 3, np.pi / 2])  # Replace with the actual value
P_min = np.array([0.5, 0.5, 0.5])  # Replace with the actual value

# Calculate the cost using the example inputs
cost = cost_function(H, DER, P, a, b, c, e, theta, P_min)
cost
