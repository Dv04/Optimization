import random


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
    while result[5] < 0:
        result = calculate_final_demand(capacity_dict)
    return result


def generate_final_demand_array(capacity_dict):
    demand_array = []
    for hour in range(1, 25):
        capacity_dict["final_demand"] = capacity_dict[f"hour_{hour}"]
        demand = generate_final_demand(capacity_dict)
        demand_array.append(demand)
    return demand_array
