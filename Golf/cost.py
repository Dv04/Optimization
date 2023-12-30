import numpy as np
from constants import capacity_dict


# Define the cost function as per the given equation.
def calculate_cost(
    P: list,
    H: int = 24,
    DER: int = 6,
    a: list = capacity_dict["A"],
    b: list = capacity_dict["B"],
    c: list = capacity_dict["C"],
    e: list = capacity_dict["D"],
    theta: list = capacity_dict["E"],
    P_min: list = capacity_dict["min_capacity"],
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
    # Assuming P now only represents one hour
    for k in range(1, DER + 1):
        F1 += ((a[k - 1] * P[k - 1] ** 2) + (b[k - 1] * P[k - 1]) + c[k - 1]) + abs(
            e[k - 1] * np.sin(theta[k - 1] * (P_min[k - 1] - P[k - 1]))
        )
    return F1
