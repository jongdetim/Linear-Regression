
import statistics as stats
import matplotlib.pyplot as plt


def predict(x, m, c):
    return m * x + c


def loss(y, x, m, c):
    return stats.mean([(value - predict(x[i], m, c)) ** 2 for i, value in enumerate(y)])


def update_plot(y_pred, i, line):
    line.set_ydata([min(y_pred), max(y_pred)])
    # line.set_xdata([min(x), max(x)])
    plt.draw()
    plt.pause(0.25 / (i + 1))


def gradient_descent(epochs, l, x, y, m, c, draw_plot, line):
    for i in range(epochs):
        c_new = l * stats.mean([predict(x[i], m, c) - value for i, value in enumerate(y)])
        m_new = l * stats.mean([(predict(x[i], m, c) - value) * x[i] for i, value in enumerate(y)])
        m, c = m - m_new, c - c_new
        y_pred = predict(x, m, c)
        # print(m, c)
        if draw_plot:
            update_plot(y_pred, i, line)
    return (m, c)


def gradient_descent_partial_deriv(epochs, l, x, y, m, c, draw_plot, line):
    n = float(len(x))
    for i in range(epochs):
        y_pred = predict(x, m, c)
        deriv_m = (-2 / n) * sum(x * (y - y_pred))
        deriv_c = (-2 / n) * sum(y - y_pred)
        m = m - l * deriv_m
        c = c - l * deriv_c
        # y_pred = predict(x, m, c)
        # print(m, c)
        if draw_plot:
            update_plot(y_pred, i, line)
    return (m, c)
