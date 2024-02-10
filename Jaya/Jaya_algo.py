import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from constants import capacity_dict
from cost import calculate_cost
from check import check_output


NUM_DER = 6
NUM_HOUR = 24
NUM_SOLUTION = 100
NUM_ITERATIONS = 100

genmat = np.zeros((NUM_HOUR, NUM_DER))
bestcost = np.zeros((NUM_HOUR, 1))
population = np.zeros((NUM_SOLUTION, NUM_DER))
population_copy = np.zeros((NUM_SOLUTION, NUM_DER))
q = np.zeros((NUM_HOUR, NUM_ITERATIONS))
mem = np.zeros((NUM_SOLUTION, NUM_DER))
ft = np.zeros(NUM_SOLUTION)
new_population = np.zeros(NUM_SOLUTION)
final_res = np.zeros((NUM_SOLUTION, NUM_DER * NUM_HOUR + 1))


for hour in range(NUM_HOUR):
    for a in range(NUM_SOLUTION):
        sixth_value = -100

        while (
            sixth_value > capacity_dict["max_capacity"][NUM_DER - 1]
            or sixth_value < capacity_dict["min_capacity"][NUM_DER - 1]
        ):
            t = 0

            for j in range(NUM_DER - 1):
                population[a, j] = (
                    np.random.rand()
                    * (
                        capacity_dict["max_capacity"][j]
                        - capacity_dict["min_capacity"][j]
                    )
                ) + capacity_dict["min_capacity"][j]
                t = t + population[a, j]
            population[a, NUM_DER - 1] = capacity_dict["hour_demand"][hour] - t
            sixth_value = population[a, NUM_DER - 1]

        new_population[a] = calculate_cost(P=population[a, :])

    mem = population

    population_copy = population.copy()

    for itr in range(NUM_ITERATIONS):
        print("Hour: ", hour, " | Iteration: ", itr)
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
                if population[i, j] < capacity_dict["min_capacity"][j]:
                    population[i, j] = capacity_dict["min_capacity"][j]
                elif population[i, j] > capacity_dict["max_capacity"][j]:
                    population[i, j] = capacity_dict["max_capacity"][j]

        for i in range(NUM_SOLUTION):
            summ = 0
            for j in range(NUM_DER - 1):
                if population[i, j] < capacity_dict["min_capacity"][j]:
                    population[i, j] = capacity_dict["min_capacity"][j]
                elif population[i, j] > capacity_dict["max_capacity"][j]:
                    population[i, j] = capacity_dict["max_capacity"][j]
                summ = summ + population[i, j]

            population[i, NUM_DER - 1] = capacity_dict["hour_demand"][hour] - summ
            if population[i, NUM_DER - 1] < capacity_dict["min_capacity"][NUM_DER - 1]:
                population[i, NUM_DER - 1] = capacity_dict["min_capacity"][NUM_DER - 1]
            elif (
                population[i, NUM_DER - 1] > capacity_dict["max_capacity"][NUM_DER - 1]
            ):
                population[i, NUM_DER - 1] = capacity_dict["max_capacity"][NUM_DER - 1]

            if np.sum(population[i, :]) != capacity_dict["hour_demand"][hour]:
                population[i, :] = population_copy[i, :].copy()

            ft[i] = calculate_cost(P=population[i, :])

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

pd.DataFrame(final_res).to_csv("data1.csv", index=False)
check_output()

print("Best iteration cost dictionary: ", bestitrcost)
final_res[:, -1] = bestitrcost
plt.plot(bestitrcost)
plt.show()
