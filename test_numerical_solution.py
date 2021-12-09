from project.gen_data import gen_data
from project.post_processing import Plotter

import matplotlib.pyplot as plt

def test_scheme(scheme_name, data_dict):
    X, y = gen_data(scheme_name, data_dict)
    Plotter(1,1).plot_2d_y(X, y)
    plt.show()


test_scheme("finite_difference_crank_nicolson", {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 30,
    "t_dim": 100,
    "a_coeff": 0.4,
    "frequency": 1,
})

test_scheme("finite_difference_euler", {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 7,
    "t_dim": 100,
    "a_coeff": 0.4,
    "frequency": 1,
})