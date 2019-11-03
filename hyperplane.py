import matplotlib.pyplot as plt
import numpy as np


def hyperplane(w,b,start,end):

    x_points = np.linspace(start, end, 2)
    y_ = -(w[0] * x_points + b) / w[1]
    plt.plot(x_points, y_)
    plt.show()
