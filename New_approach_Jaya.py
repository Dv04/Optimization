import numpy as np
import matplotlib.pyplot as plt
from cost import calculate_cost as forpapercostfun

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
NUM_SOLUTION = 100
population = np.zeros((NUM_SOLUTION, NUM_HOUR * NUM_DER))
population_copy = np.zeros((NUM_SOLUTION, NUM_HOUR * NUM_DER))
NUM_ITERATIONS = 100
q = np.zeros((NUM_HOUR, NUM_ITERATIONS))
max_limit = [200, 80, 50, 35, 30, 40]
min_limit = [50, 20, 15, 10, 10, 12]
mem = np.zeros((NUM_SOLUTION, NUM_DER))
ft = np.zeros(NUM_SOLUTION)
new_population = np.zeros(NUM_SOLUTION)

for hour in range(NUM_HOUR):
    # print(hour)
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
    print(population)

    new_population[a] = forpapercostfun(P=population[a, :])

    mem = population
    min_cost, min_index = np.min(new_population), np.argmin(new_population)
    bestpop = mem[min_index, :]

    population_copy = population
    for itr in range(NUM_ITERATIONS):
        min_cost, min_index = np.min(new_population), np.argmin(new_population)
        bestpop = min_index
        max_cost, max_index = np.max(new_population), np.argmax(new_population)
        worstpop = max_index

        npum = np.ceil(NUM_SOLUTION * np.random.rand(1, NUM_SOLUTION))
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

            population[i, NUM_DER] = L[hour] - summ

            if population[i, NUM_DER] < min_limit[NUM_DER]:
                population[i, NUM_DER] = min_limit[NUM_DER]
            elif population[i, NUM_DER] > max_limit[NUM_DER]:
                population[i, NUM_DER] = max_limit[NUM_DER]

            if np.sum(population[i, :]) != L[hour]:
                population[i, :] = population_copy[i, :]

            ft[i] = forpapercostfun(population=population[i, :])

        population_copy = population
        for i in range(NUM_SOLUTION):
            if ft[i] < new_population[i]:
                mem[i, :] = population[i, :]
                new_population[i] = ft[i]

        min_cost, min_index = np.min(new_population), np.argmin(new_population)
        bestpop = mem[min_index, :]
        bestcost[hour] = min_cost
        genmat[hour, :] = population[min_index, :]
        q[hour, itr] = min_cost

totalcost = np.sum(bestcost)
tch = np.zeros((NUM_HOUR, 1))
for hour in range(NUM_HOUR):
    tch[hour] = np.sum(genmat[hour, :])

result = [genmat, tch, bestcost]
bestitrcost = np.zeros(NUM_ITERATIONS)
for u in range(NUM_ITERATIONS):
    bestitrcost[u] = np.sum(q[:, u])

plt.plot(bestitrcost)
plt.show()
