from sklearn.metrics import pairwise_distances
import numpy as np


def do_stuff(data, y_1, y_2):
    dist_y_1 = pairwise_distances(data, y_1)
    dist_y_2 = pairwise_distances(data, y_2)
    bool_filter = np.ones(len(data), dtype=bool)

    for i in range(len(data)):
        bool_filter[i] = dist_y_1[i, 0] < dist_y_2[i, 0]

    bool_filter_neg = [not val for val in bool_filter]
    y_1_data = data[bool_filter]
    y_2_data = data[bool_filter_neg]
    new_y_1 = [[sum(y_1_data[:, 0]) / len(y_1_data), sum(y_1_data[:, 1]) / len(y_1_data)]]
    new_y_2 = [[sum(y_2_data[:, 0]) / len(y_2_data), sum(y_2_data[:, 1]) / len(y_2_data)]]
    if np.array_equal(new_y_1, y_1) and np.array_equal(new_y_2, y_2):
        print(y_1)
        print(y_1_data)
        print(y_2)
        print(y_2_data)
    else:
        do_stuff(data, new_y_1, new_y_2)


def main():
    data = np.array([[1.7, 1.5], [1.3, 1.8], [1.9, 2.2], [2.6, 2.3], [3.4, 2.1], [3.8, 2.6]])
    y_1 = [[2.0, 2.5]]
    y_2 = [[2.6, 1.7]]
    do_stuff(data, y_1, y_2)


main()
