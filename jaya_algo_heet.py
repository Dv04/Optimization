# Implementing the Jaya algorithm for the provided optimization problem
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from constants import capacity_dict
from cost import calculate_cost
from calculate import calculate_final_demand
from check import check_output

# Constants and initial setup
NUM_SOLUTIONS = 10  # Population size
NUM_ITERATIONS = 5
NO_CHANGE_THRESHOLD = 4  # Terminate if no significant change for this many iterations
NUM_HOURS = 24
NUM_DER = 6
NUM_COLUMNS = NUM_HOURS * NUM_DER + 1  # 144 for power values, 1 for cost


def generate_final_demand(capacity_dict, hour):
    capacity_dict["final_demand"] = capacity_dict["hour_demand"][hour]
    result = calculate_final_demand(capacity_dict)
    while not (
        (
            capacity_dict["min_capacity"][0]
            <= result[0]
            <= capacity_dict["max_capacity"][0]
        )
        and (
            capacity_dict["min_capacity"][1]
            <= result[1]
            <= capacity_dict["max_capacity"][1]
        )
        and (
            capacity_dict["min_capacity"][2]
            <= result[2]
            <= capacity_dict["max_capacity"][2]
        )
        and (
            capacity_dict["min_capacity"][3]
            <= result[3]
            <= capacity_dict["max_capacity"][3]
        )
        and (
            capacity_dict["min_capacity"][4]
            <= result[4]
            <= capacity_dict["max_capacity"][4]
        )
        and (
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
    # Initialize population
    population = generate_initial_population()
    
    min_iter = {}  # To track cost at each iteration for plotting

    def number_generator():
        r1, r2 = random.random(), random.random()
        return r1, r2


    def num_gen(j, r1, r2):
        new_solution[j] += r1 * (best_solution[j] - abs(new_solution[j])) - r2 * (
            worst_solution[j] - abs(new_solution[j])
        )

    def regenerate_previous_values():
        r1, r2 = number_generator()
        print("index:", i, r1, r2)
        for j in range(NUM_DER * NUM_HOURS):
            num_gen(j, r1, r2)

    for iteration in range(NUM_ITERATIONS):
        best_solution = population[np.argmin(population[:, -1])]
        worst_solution = population[np.argmax(population[:, -1])]

        # Update each solution
        for i in range(NUM_SOLUTIONS):
            new_solution = np.copy(population[i])

            while True:
                regenerate_previous_values()

                for j in range(NUM_DER * NUM_HOURS):
                    unit_index = j % NUM_DER

                    if not j % NUM_DER == NUM_DER - 1:
                        if capacity_dict["min_capacity"][unit_index] <= new_solution[j] <= capacity_dict["max_capacity"][unit_index]:
                            pass
                        else:
                            # Constraint checking: Ensure values are within min and max capacity
                            new_solution[j] = max(
                                capacity_dict["min_capacity"][unit_index], new_solution[j]
                            )
                            new_solution[j] = min(
                                capacity_dict["max_capacity"][unit_index], new_solution[j]
                            )
                    else:
                        new_solution[j] = capacity_dict["hour_demand"][j // NUM_DER] - np.sum(new_solution[j - NUM_DER + 1 : j])

                if all(
                    capacity_dict["min_capacity"][unit_index]
                    <= new_solution[5]
                    <= capacity_dict["max_capacity"][unit_index]
                    for j in range(5, NUM_DER * NUM_HOURS, 6)
                ):
                    print("This is what you want")
                    break  # Exit the loop if the entire row satisfies the condition

                print("Regenerating entire row: Iteration", iteration, "Index:", i)

            # Update the cost for the new solution
            new_solution[-1], _ = calculate_cost(
                H=NUM_HOURS,
                DER=NUM_DER,
                P=new_solution[:-1],
                a=capacity_dict["A"],
                b=capacity_dict["B"],
                c=capacity_dict["C"],
                e=capacity_dict["D"],
                theta=capacity_dict["E"],
                P_min=capacity_dict["min_capacity"],
            )

            # Replace the old solution if the new one is better
            if new_solution[-1] < population[i, -1]:
                population[i] = new_solution

        # Log and plot progress
        current_best_cost = np.min(population[:, -1])
        min_iter[iteration] = current_best_cost
        print("Iteration", iteration, "completed with best cost:", current_best_cost)

    # Plot the cost vs iteration graph
    plt.plot(list(min_iter.keys()), list(min_iter.values()))
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost vs Iteration")
    plt.savefig("cost_vs_iteration.png")
    plt.show()

    # Return the best solution found
    return population[np.argmin(population[:, -1])]


# Running the Jaya algorithm
best_solution = jaya_algorithm()
# print("Best Solution:", best_solution)

check_output()

# Save the best solution to a file
pd.DataFrame(best_solution.reshape(1, -1)).to_csv("best_solution_new.csv", index=False)
