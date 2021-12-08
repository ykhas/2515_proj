from project.gen_data import gen_data
from project.post_processing import Plotter

import matplotlib.pyplot as plt

data_dict = {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 100,
    "t_dim": 100,
    "a_coeff": 0.4,
    "frequency": 1,
}

X, y = gen_data("heat_1d_boundary_sin_exact", data_dict)
Plotter(1,1).plot_2d_y(X, y)
plt.show()
