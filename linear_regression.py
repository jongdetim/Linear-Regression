import numpy as np
import matplotlib.pyplot as plt


def predict(x, m, c):
    return m * x + c


def loss(y, x, m, c):
    return np.mean([(value - predict(x[i], m, c)) ** 2 for i, value in enumerate(y)])


def standardize_data(x_orig, y_orig):
    x = (x_orig - np.mean(x_orig)) / np.std(x_orig)
    # y = (y_orig - np.mean(y_orig)) / np.std(y_orig)
    return x, y_orig


# def destandardize_data(data, x_orig, y_orig):
#     x = (x * np.std(x_orig)) + np.mean(x_orig)
#     y = (y * np.std(y_orig)) + np.mean(y_orig)
#     return x, y

# def get_thetas_denormalized(x, y, m, c):
#     standardized_pred = predict(x, m, c)
#     pred = destandardize_data(standardized_pred)
#     x, y = destandardize_data(x, y, x_orig, y_orig)

def update_plot(y_pred, i, line, x_orig, m, c):
    m_old = m
    m = m / np.std(x_orig)
    c = c - (m_old * np.mean(x_orig)) / np.std(x_orig)
    y_pred = predict(x_orig, m, c)
    if m < 0:
        line.set_ydata([max(y_pred), min(y_pred)])
        # line.set_xdata([max(x), min(x)])
    else:
        line.set_ydata([min(y_pred), max(y_pred)])
        # line.set_xdata([min(x), max(x)])
    plt.draw()
    plt.pause(0.25 / (i + 1))


def gradient_descent(epochs, l, x, y, m, c, draw_plot, line, x_orig):
    n = float(len(x))
    for i in range(epochs):
        c_new = l * (1 / n) * sum([predict(x[i], m, c) - value for i, value in enumerate(y)])
        m_new = l * (1 / n) * sum([(predict(x[i], m, c) - value) * x[i] for i, value in enumerate(y)])
        m, c = m - m_new, c - c_new
        y_pred = predict(x, m, c)
        print(m, c)
        if draw_plot:
            update_plot(y_pred, i, line, x_orig, m, c)
    return (m, c)


def gradient_descent_partial_deriv(epochs, l, x, y, m, c, draw_plot, line):
    n = float(len(x))
    for i in range(epochs):
        y_pred = predict(x, m, c)
        deriv_m = (-2 / n) * sum(x * (y - y_pred))
        deriv_c = (-2 / n) * sum(y - y_pred)
        m = m - l * deriv_m
        c = c - l * deriv_c
        if draw_plot:
            update_plot(y_pred, i, line, x, m)
    return (m, c)
