import numpy as np
import matplotlib.pyplot as plt


class linear_regression:

    def __init__(self, m=0.0, c=0.0):
        self.m = m
        self.c = c
        self.x_orig = None
        self.plt = None

    def predict(self, x, m, c):
        return m * x + c

    def loss(self, y, x, m, c):
        return np.mean([(value - self.predict(x[i], m, c)) ** 2 for i, value in enumerate(y)])

    def standardize_data(self, x_orig):
        if self.x_orig is None:
            self.x_orig = x_orig
        x_standard = (self.x_orig - np.mean(self.x_orig)) / np.std(self.x_orig)
        return x_standard

    def get_destandardized_thetas(self):
        if self.x_orig is None:
            print("no original data known. please use model.fit() first")
            return
        c_destandard = self.c - (self.m * np.mean(self.x_orig)) / np.std(self.x_orig)
        m_destandard = self.m / np.std(self.x_orig)
        return m_destandard, c_destandard

    def update_plot(self, y_pred, i, line):
        m_destandard, c_destandard = self.get_destandardized_thetas()
        y_pred = self.predict(self.x_orig, m_destandard, c_destandard)
        if self.m < 0:
            line.set_ydata([max(y_pred), min(y_pred)])
        else:
            line.set_ydata([min(y_pred), max(y_pred)])
        self.plt.draw()
        self.plt.pause(0.25 / (i + 1))

    def gradient_descent(self, epochs, l, x, y, draw_plot, line):
        n = float(len(x))
        for i in range(epochs):
            c_new = l * (1 / n) * sum([self.predict(x[i], self.m, self.c) - value for i, value in enumerate(y)])
            m_new = l * (1 / n) * sum([(self.predict(x[i], self.m, self.c) - value) * x[i] for i, value in enumerate(y)])
            self.m, self.c = self.m - m_new, self.c - c_new
            y_pred = self.predict(x, self.m, self.c)
            if draw_plot:
                self.update_plot(y_pred, i, line)
        return (self.m, self.c)

    def gradient_descent_partial_deriv(self, epochs, l, x, y, draw_plot, line):
        n = float(len(x))
        for i in range(epochs):
            y_pred = self.predict(x, self.m, self.c)
            deriv_m = (-2 / n) * sum(x * (y - y_pred))
            deriv_c = (-2 / n) * sum(y - y_pred)
            m = m - l * deriv_m
            c = c - l * deriv_c
            if draw_plot:
                self.update_plot(y_pred, i, line)
        return (m, c)

    def fit(self, x, y, epochs=1000, l=0.025, draw_plot=False, line=None, plt=None):
        self.plt = plt
        if draw_plot:
            plt.ion()
            plt.scatter(self.x_orig, y)
            y_pred = self.predict(self.x_orig, self.m, self.c)
            (line,) = plt.plot([min(self.x_orig), max(self.x_orig)], [min(y_pred), max(y_pred)], color="red")
            line.axes.set_xlim(min(self.x_orig) - (max(self.x_orig) * 0.05), max(self.x_orig) * 1.05)
            line.axes.set_ylim(min(y) - (max(y) * 0.05), max(y) * 1.05)
            plt.draw()
            plt.pause(0.01)
        self.gradient_descent(epochs, l, x, y, draw_plot, line)
