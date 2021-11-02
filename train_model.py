#!/usr/bin/python3

from linear_regression import *
import sys
import numpy as np

if __name__ == "__main__":

    try:
        data = np.genfromtxt("data.csv", delimiter=',', dtype=int)
        x_orig = data[1:,0]
        y_orig = data[1:,1]
    except:
        print("no datafile present. please provide data as 'data.csv' in this directory")
        exit(1)

    x, y = standardize_data(x_orig, y_orig)

    print(x, y)

    m, c = 0.0, np.mean(y)
    l = 0.025
    epochs = 2000
    line = None
    draw_plot = False
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        draw_plot = True

    if draw_plot:
        plt.ion()
        plt.scatter(x_orig, y)
        y_pred = predict(x_orig, m, c)
        (line,) = plt.plot([min(x_orig), max(x_orig)], [min(y_pred), max(y_pred)], color="red")
        line.axes.set_xlim(min(x_orig) - (max(x_orig) * 0.05), max(x_orig) * 1.05)
        line.axes.set_ylim(min(y) - (max(y) * 0.05), max(y) * 1.05)
        plt.draw()
        plt.pause(0.2)

    m, c = gradient_descent(epochs, l, x, y, m, c, draw_plot, line, x_orig)
    # m, c = get_thetas_denormalized(x, y, m, c)
    m_old = m
    m = m / np.std(x_orig)
    c = c - (m_old * np.mean(x_orig)) / np.std(x_orig)

    print("m: ", m, "\nc: ", c)
    print("loss: ", loss(y_orig, x_orig, m, c))
    with open("thetas.txt", "w+") as file:
        file.write(str(m) + "\n" + str(c))

    if draw_plot:
        plt.ioff()
        plt.show()
