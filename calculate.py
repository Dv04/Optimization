import random
import numpy as np
from constants import capacity_dict
from cost import calculate_cost


def calculate_final_demand(capacity_dict):
    var1 = (
        capacity_dict["max_capacity"][0] - capacity_dict["min_capacity"][0]
    ) * random.random() + capacity_dict["min_capacity"][0]
    var2 = (
        capacity_dict["max_capacity"][1] - capacity_dict["min_capacity"][1]
    ) * random.random() + capacity_dict["min_capacity"][1]
    var3 = (
        capacity_dict["max_capacity"][2] - capacity_dict["min_capacity"][2]
    ) * random.random() + capacity_dict["min_capacity"][2]
    var4 = (
        capacity_dict["max_capacity"][3] - capacity_dict["min_capacity"][3]
    ) * random.random() + capacity_dict["min_capacity"][3]
    var5 = (
        capacity_dict["max_capacity"][4] - capacity_dict["min_capacity"][4]
    ) * random.random() + capacity_dict["min_capacity"][4]

    var6 = capacity_dict["final_demand"] - (var1 + var2 + var3 + var4 + var5)

    return [var1, var2, var3, var4, var5, var6]


def generate_final_demand(capacity_dict):
    result = calculate_final_demand(capacity_dict)
    while not (
        capacity_dict["min_capacity"][5] < result[5] < capacity_dict["max_capacity"][4]
    ):
        result = calculate_final_demand(capacity_dict)
    return result


def generate_final_demand_array(capacity_dict):
    demand_array = np.empty((100, 145))
    for row in range(100):
        for hour in range(0, 24):
            capacity_dict["final_demand"] = capacity_dict["hour_demand"][hour]
            demand = generate_final_demand(capacity_dict)
            demand_array[row, (hour) * 6 : (hour + 1) * 6] = demand[:6]
            demand_array[row, -1] = calculate_cost(
                H=24,
                DER=6,
                P=demand_array[row, :-1],
                a=capacity_dict["A"],
                b=capacity_dict["B"],
                c=capacity_dict["C"],
                e=capacity_dict["E"],
                theta=capacity_dict["D"],
                P_min=capacity_dict["min_capacity"],
            )

    return demand_array


final_array = generate_final_demand_array(capacity_dict)
print(final_array)

with open("result.csv", "w") as f:
    for row in final_array:
        f.write("%s\n" % row)
