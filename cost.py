import numpy as np


# Define the cost function as per the given equation.
def calculate_cost(
    H: int,
    DER: int,
    P: list,
    a: list,
    b: list,
    c: list,
    e: list,
    theta: list,
    P_min: list,
) -> float:
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
    for t in range(1, H + 1):
        for k in range(1, DER):
            F1 += (
                (a[k] * P[(t - 1) * 6 + k] ** 2) + (b[k] * P[(t - 1) * 6 + k]) + c[k]
            ) + (e[k] * np.sin(theta[k]) * (P_min[k] - P[(t - 1) * 6 + k]))
    return F1
