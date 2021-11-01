#!/usr/bin/python3

import numpy as np
import statistics as stats

x = np.array([1, 2, 3])
y = np.array([6, 12, 18])


def predict(x, m, c):
    return m * x + c


def loss(y, x, m, c):
    return stats.mean([(value - predict(x[i], m, c)) ** 2 for i, value in enumerate(y)])


print(loss(y, x, 6, 0))
