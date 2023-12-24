import numpy as np


# Define the cost function as per the given equation.
def calculate_cost_temp(
    H: int,
    DER: int,
    P: list,
) -> float:
    # Perform the double summation
    for t in range(1, H + 1):
        for k in range(1, DER + 1):
            print((t - 1) * 6 + k - 1)
