import numpy as np
import random
import pandas as pd
import copy
import matplotlib.pyplot as plt
from constants import capacity_dict
from cost import calculate_cost
from calculate import calculate_final_demand
from check import check_output

# Constants and initial setup
NUM_SOLUTIONS = 400
NUM_ITERATIONS = 1000
NUM_HOURS = 24
NUM_DER = 6
NUM_COLUMNS = NUM_HOURS * NUM_DER + 1


def generate_final_demand(capacity_dict, hour):
    capacity_dict["final_demand"] = capacity_dict["hour_demand"][hour]
    result = calculate_final_demand(capacity_dict)
    while not (
        (
            capacity_dict["min_capacity"][5]
            <= result[5]
            <= capacity_dict["max_capacity"][5]
        )
    ):
        result = calculate_final_demand(capacity_dict)
    return result


def generate_initial_population():
    population = np.zeros((NUM_SOLUTIONS, NUM_COLUMNS))
    for i in range(NUM_SOLUTIONS):
        for hour in range(NUM_HOURS):
            # Generate demand satisfying the sum constraint
            demand = generate_final_demand(capacity_dict, hour)
            population[i, hour * NUM_DER : (hour + 1) * NUM_DER] = demand[:NUM_DER]
        # Calculate cost for each solution
        population[i, -1], _ = calculate_cost(
            H=NUM_HOURS,
            DER=NUM_DER,
            P=population[i, :-1],
            a=capacity_dict["A"],
            b=capacity_dict["B"],
            c=capacity_dict["C"],
            e=capacity_dict["D"],
            theta=capacity_dict["E"],
            P_min=capacity_dict["min_capacity"],
        )
    return population


def jaya_algorithm():
    population = generate_initial_population()
    min_iter = {}

    for iteration in range(NUM_ITERATIONS):
        print(f"Starting iteration {iteration}...")
        best_solution = population[np.argmin(population[:, -1])]
        worst_solution = population[np.argmax(population[:, -1])]

        for i in range(NUM_SOLUTIONS):
            new_solution = np.copy(population[i])
            r1, r2 = random.random(), random.random()

            for j in range(NUM_DER * NUM_HOURS):
                unit_index = j % NUM_DER
                if not j % NUM_DER == NUM_DER - 1:
                    new_solution[j] += r1 * (
                        best_solution[j] - abs(new_solution[j])
                    ) - r2 * (worst_solution[j] - abs(new_solution[j]))

            for j in range(NUM_DER * NUM_HOURS):
                unit_index = j % NUM_DER
                if not (j % NUM_DER == NUM_DER - 1):
                    new_solution[j] = max(
                        capacity_dict["min_capacity"][unit_index], new_solution[j]
                    )
                    new_solution[j] = min(
                        capacity_dict["max_capacity"][unit_index], new_solution[j]
                    )
                else:
                    new_solution[j] = capacity_dict["hour_demand"][
                        j // NUM_DER
                    ] - np.sum(new_solution[j - NUM_DER + 1 : j])

            valid = False
            initial_solution = copy.deepcopy(new_solution)

            while not valid:
                # Reset new_solution to its initial state at the beginning of each iteration
                new_solution = copy.deepcopy(initial_solution)

                for j in range(NUM_DER * NUM_HOURS):
                    unit_index = j % NUM_DER

                    if not (j % NUM_DER == NUM_DER - 1):
                        new_solution[j] += r1 * (
                            best_solution[j] - abs(new_solution[j])
                        ) - r2 * (worst_solution[j] - abs(new_solution[j]))

                        new_solution[j] = max(
                            capacity_dict["min_capacity"][unit_index], new_solution[j]
                        )
                        new_solution[j] = min(
                            capacity_dict["max_capacity"][unit_index], new_solution[j]
                        )
                    else:
                        new_solution[j] = capacity_dict["hour_demand"][
                            j // NUM_DER
                        ] - np.sum(new_solution[j - NUM_DER + 1 : j])

                valid = np.logical_and.reduce(
                    capacity_dict["min_capacity"][5]
                    <= new_solution[k]
                    <= capacity_dict["max_capacity"][5]
                    for k in range(5, NUM_DER * NUM_HOURS, 6)
                )

            new_solution[-1], _ = calculate_cost(P=new_solution[:-1])

            if new_solution[-1] < population[i, -1]:
                population[i] = new_solution

        current_best_cost = np.min(population[:, -1])
        min_iter[iteration] = current_best_cost
        print(
            "Regenerating entire row: Iteration",
            iteration,
            "Best cost:",
            current_best_cost,
        )
    plt.plot(list(min_iter.keys()), list(min_iter.values()))
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration")
    plt.savefig("cost_vs_iteration.png")
    plt.show()

    return population[np.argmin(population[:, -1])]


best_solution = jaya_algorithm()
pd.DataFrame(best_solution.reshape(1, -1)).to_csv("best_solution_new.csv", index=False)
check_output()
