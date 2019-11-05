import matplotlib.pyplot as plt
import numpy as np


def hyperplane(w,b,start,end):

    x_points = np.linspace(start, end, 2)
    y_ = -(w[0] * x_points + b) / w[1]
    y_plus = -(w[0] * x_points + b + 1) / w[1]
    y_sub = -(w[0] * x_points + b - 1) / w[1]
    plt.plot(x_points, y_)
    plt.plot(x_points, y_plus, linewidth=2, linestyle="--")
    plt.plot(x_points, y_sub, linewidth=2, linestyle="--")
    plt.show()
