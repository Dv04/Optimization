# open csv file and print sum of every 6 columns of second row
import csv
import numpy as np
from constants import capacity_dict


def check_output():
    with open("best_solution_new.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        data = list(reader)
        data = np.array(data[1])
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
