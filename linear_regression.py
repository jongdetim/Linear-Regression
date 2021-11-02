#!/usr/bin/python3

import numpy as np
import statistics as stats
import matplotlib.pyplot as plt


def predict(x, m, c):
    return m * x + c


def loss(y, x, m, c):
    return stats.mean([(value - predict(x[i], m, c)) ** 2 for i, value in enumerate(y)])


def update_plot(y_pred):
    line.set_ydata([min(y_pred), max(y_pred)])
    # line.set_xdata([min(x), max(x)])
    plt.draw()


def gradient_descent(epochs, l, x, y, m, c, n, draw_plot):
    for _ in range(epochs):
        y_pred = predict(x, m, c)
        deriv_m = (-2 / n) * sum(x * (y - y_pred))
        deriv_c = (-2 / n) * sum(y - y_pred)
        m = m - l * deriv_m
        # c_new = l * stats.mean([predict(x[i], m, c) - value for i, value in enumerate(y)])
        c = c - l * deriv_c
        # m_new = l * stats.mean([(predict(x[i], m, c) - value) * x[i] for i, value in enumerate(y)])
        # m, c = m + m_new, c + c_new
        # y_pred = predict(x, m, c)
        print(m, c)
        if draw_plot:
            update_plot(y_pred)
            plt.pause(5 / epochs)
    return (m, c)


if __name__ == "__main__":

    x = np.array([1, 2, 3, 4, 3, 1, 6, 8, 4, 2])
    y = np.array([6, 12, 18, 54, 32, 1, 23, 2, 6, 3])
    n = float(len(x))
    m, c = 1, 0
    l = 0.01
    epochs = 100
    draw_plot = True

    if draw_plot:
        plt.ion()
        plt.scatter(x, y)
        y_pred = predict(x, m, c)
        (line,) = plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color="red")
        line.axes.set_xlim(0, max(x) + 1)
        line.axes.set_ylim(0, max(y) + 1)
        plt.draw()

    m, c = gradient_descent(epochs, l, x, y, m, c, n, draw_plot)
    print("m: ", m, "\nc: ", c)
    print("loss: ", loss(y, x, m, c))

    if draw_plot:
        plt.ioff()
        plt.show()
