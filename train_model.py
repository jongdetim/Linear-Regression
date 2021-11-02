#!/usr/bin/python3

from linear_regression import *
import sys
import numpy as np

if __name__ == "__main__":

    x = np.array([1, 2, 3, 4, 3, 1, 6, 8, 4, 2])
    y = np.array([6, 12, 18, 54, 32, 1, 23, 2, 6, 3])
    m, c = 1, 0
    l = 0.05
    epochs = 100
    line = None
    draw_plot = False
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        draw_plot = True

    if draw_plot:
        plt.ion()
        plt.scatter(x, y)
        y_pred = predict(x, m, c)
        (line,) = plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color="red")
        line.axes.set_xlim(0, max(x) + 1)
        line.axes.set_ylim(0, max(y) + 1)
        plt.draw()

    m, c = gradient_descent(epochs, l, x, y, m, c, draw_plot, line)
    print("m: ", m, "\nc: ", c)
    print("loss: ", loss(y, x, m, c))
    with open("thetas.txt", "w+") as file:
        file.write(str(m) + "\n" + str(c))

    if draw_plot:
        plt.ioff()
        plt.show()
