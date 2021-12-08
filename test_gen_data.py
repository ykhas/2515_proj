from project.gen_data import gen_data
from project.post_processing import plot_2d_y

import matplotlib.pyplot as plt

x_range = (0, 1)
t_range = (0, 1)
x_dim = 200
t_dim = 200
a_coef = 0.4
freq = 1

data_dict = {
    "x_range": (0, 1),
    "t_range": (0, 1),
    "x_dim": 200,
    "t_dim": 200,
    "a_coeff": 0.4,
    "frequency": 1,
}

X, y = gen_data("heat_1d_boundary_sin_exact", data_dict)
plot_2d_y(X, y)
plt.show()
