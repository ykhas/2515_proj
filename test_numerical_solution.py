from project.gen_data import gen_data
from project.post_processing import Plotter

import matplotlib.pyplot as plt

data_dict = {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 7,
    "t_dim": 100,
    "a_coeff": 0.4,
    "frequency": 1,
}

# this solution requires the number of time steps to be extremely large
# see https://en.wikipedia.org/wiki/Von_Neumann_stability_analysis for reasoning.
X, y = gen_data("finite_difference_euler", data_dict)
Plotter(1,1).plot_2d_y(X, y)
plt.show()
