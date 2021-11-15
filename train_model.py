#!/usr/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from linear_regression import linear_regression


def read_file():
    try:
        data = np.genfromtxt("data.csv", delimiter=',', dtype=int)
        return data[1:, 0], data[1:, 1]
    except:
        print("no datafile present. please provide data as 'data.csv' with 2 columns and headers in this directory")
        exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="train a linear regression model using gradient descent on a provided dataset")
    parser.add_argument("--visualize", "-v", help="find a path as fast as possible, not guaranteed to be shortest solution", action='store_true')
    args = parser.parse_args()
    return args


def main():
    l = 0.025
    epochs = 1000
    line = None
    args = parse_args()
    x_orig, y = read_file()

    model = linear_regression()
    x = model.standardize_data(x_orig)
    model.fit(x, y, epochs, l, args.visualize, line, plt)
    m, c = model.get_destandardized_thetas()

    print("m: ", m, "\nc: ", c)
    print("loss: ", model.loss(y, x_orig, m, c))
    with open("thetas.txt", "w+") as file:
        file.write(str(m) + "\n" + str(c))

    if args.visualize:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
