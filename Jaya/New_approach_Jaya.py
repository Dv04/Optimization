import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

capacity_dict = {
    "max_capacity": [200, 80, 50, 35, 30, 40],
    "min_capacity": [50, 20, 15, 10, 10, 12],
    "hour_demand": [
        166,
        196,
        229,
        267,
        283.4,
        272,
        246,
        213,
        192,
        161,
        147,
        160,
        170,
        185,
        208,
        232,
        246,
        241,
        236,
        225,
        204,
        182,
        161,
        131,
    ],
    "A": [0.00375, 0.0175, 0.0625, 0.00834, 0.025, 0.025],
    "B": [2, 1.75, 1, 3.25, 3, 3],
    "C": [0, 0, 0, 0, 0, 0],
    "D": [18, 16, 14, 12, 13, 13.5],
    "E": [0.037, 0.038, 0.04, 0.045, 0.042, 0.041],
}


def check_output():
    with open("data1.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = list(reader)
        data = np.array(data[-1])
        data = list(map(float, data))

        falseCtn = 0

        for i in range(len(data) - 1):
            if (
                (capacity_dict["min_capacity"][i % 6])
                <= data[i]
                <= capacity_dict["max_capacity"][i % 6]
            ) and data[i] > 0:
                print(
                    "\033[92m" + "True for index: ",
                    i,
                    " | ",
                    data[i],
                    " | Wanted: [",
                    (capacity_dict["min_capacity"][i % 6]),
                    ",",
                    capacity_dict["max_capacity"][i % 6],
                    "]" + "\033[0m",
                )
            else:
                print(
                    "False for index: ",
                    i,
                    " | ",
                    data[i],
                    " | Wanted: [",
                    (capacity_dict["min_capacity"][i % 6]),
                    ",",
                    capacity_dict["max_capacity"][i % 6],
                    "]",
                )
                falseCtn += 1

        if falseCtn == 0:
            # make the print statement in color
            print("\033[92m" + "\n\nAll values are within bounds.\n\n" + "\033[0m")

            # print("\n\nAll values are within bounds.\n\n")
        else:
            print("Values out of bounds: ", falseCtn)

        ctn = 0
        for i in range(0, len(data) - 1, 6):
            temp = data[i : i + 6]
            if sum(map(float, temp)) == capacity_dict["hour_demand"][i // 6]:
                ctn += 1

            print(
                "New solution sum: ",
                sum(map(float, temp)),
                "| Wanted: ",
                capacity_dict["hour_demand"][i // 6],
                " ",
            )

        print("\n\nHour demand matched: ", ctn, "times.")


# Define the cost function as per the given equation.
def forpapercostfun(
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


L = [
    166,
    196,
    229,
    267,
    283.4,
    272,
    246,
    213,
    192,
    161,
    147,
    160,
    170,
    185,
    208,
    232,
    246,
    241,
    236,
    225,
    204,
    182,
    161,
    131,
]

NUM_DER = 6
NUM_HOUR = 24
genmat = np.zeros((NUM_HOUR, NUM_DER))
bestcost = np.zeros((NUM_HOUR, 1))
NUM_SOLUTION = 500
population = np.zeros((NUM_SOLUTION, NUM_DER))
population_copy = np.zeros((NUM_SOLUTION, NUM_DER))
NUM_ITERATIONS = 1000
q = np.zeros((NUM_HOUR, NUM_ITERATIONS))
max_limit = [200, 80, 50, 35, 30, 40]
min_limit = [50, 20, 15, 10, 10, 12]
mem = np.zeros((NUM_SOLUTION, NUM_DER))
ft = np.zeros(NUM_SOLUTION)
new_population = np.zeros(NUM_SOLUTION)

final_res = np.zeros((NUM_SOLUTION, NUM_DER * NUM_HOUR + 1))

for hour in range(NUM_HOUR):
    for a in range(NUM_SOLUTION):
        sixth_value = -100

        while (
            sixth_value > max_limit[NUM_DER - 1] or sixth_value < min_limit[NUM_DER - 1]
        ):
            t = 0

            for j in range(NUM_DER - 1):
                population[a, j] = (
                    np.random.rand() * (max_limit[j] - min_limit[j])
                ) + min_limit[j]
                t = t + population[a, j]
            population[a, NUM_DER - 1] = L[hour] - t
            sixth_value = population[a, NUM_DER - 1]

        new_population[a] = forpapercostfun(P=population[a, :])

    mem = population

    population_copy = population.copy()

    for itr in range(NUM_ITERATIONS):
        print(hour, itr)
        min_cost, min_index = np.min(new_population), np.argmin(new_population)
        with open("data1.txt", "w") as f:
            f.write(str(new_population) + "\n" + str(min_cost) + "\n" + str(min_index))

        bestpop = min_index
        max_cost, max_index = np.max(new_population), np.argmax(new_population)
        worstpop = max_index

        for i in range(NUM_SOLUTION):
            for j in range(NUM_DER):
                r1 = np.random.rand()
                r2 = np.random.rand()

                population[i, j] = (
                    population[i, j]
                    + r1 * (mem[bestpop, j] - population[i, j])
                    - r2 * (mem[worstpop, j] - population[i, j])
                )

        for i in range(NUM_SOLUTION):
            for j in range(NUM_DER - 1):
                if population[i, j] < min_limit[j]:
                    population[i, j] = min_limit[j]
                elif population[i, j] > max_limit[j]:
                    population[i, j] = max_limit[j]

        for i in range(NUM_SOLUTION):
            summ = 0
            for j in range(NUM_DER - 1):
                if population[i, j] < min_limit[j]:
                    population[i, j] = min_limit[j]
                elif population[i, j] > max_limit[j]:
                    population[i, j] = max_limit[j]
                summ = summ + population[i, j]

            population[i, NUM_DER - 1] = L[hour] - summ
            if population[i, NUM_DER - 1] < min_limit[NUM_DER - 1]:
                population[i, NUM_DER - 1] = min_limit[NUM_DER - 1]
            elif population[i, NUM_DER - 1] > max_limit[NUM_DER - 1]:
                population[i, NUM_DER - 1] = max_limit[NUM_DER - 1]

            if np.sum(population[i, :]) != L[hour]:
                population[i, :] = population_copy[i, :].copy()

            ft[i] = forpapercostfun(P=population[i, :])

        population_copy = population.copy()

        for i in range(NUM_SOLUTION):
            if ft[i] < new_population[i]:
                mem[i, :] = population[i, :].copy()

                new_population[i] = ft[i]

        min_cost, min_index = np.min(new_population), np.argmin(new_population)
        bestpop = mem[min_index, :]

        bestcost[hour] = min_cost

        genmat[hour, :] = population[min_index, :].copy()
        q[hour, itr] = min_cost

    final_res[:, hour * 6 : (hour + 1) * 6] = population


totalcost = np.sum(bestcost)
tch = np.zeros((NUM_HOUR, 1))
for hour in range(NUM_HOUR):
    tch[hour] = np.sum(genmat[hour, :])

result = [genmat, tch, bestcost]
bestitrcost = np.zeros(NUM_ITERATIONS)
for u in range(NUM_ITERATIONS):
    bestitrcost[u] = np.sum(q[:, u])

check_output()
print(bestitrcost)
final_res[:, -1] = bestitrcost
pd.DataFrame(final_res).to_csv("data1.csv", index=False)
plt.plot(bestitrcost)
plt.show()
