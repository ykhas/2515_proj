import numpy as np

def gen_data(func_name = "", dict_args = {}):
    """
    Returns the grid X, and solution y, by n data points
    X.shape (n, 2)
    y.shape (n, 1)

    Parameters
    ----------
    func_name : str
    """

    if func_name == "":
        return None, None

    return globals()[func_name](dict_args)


def create_t_x_grids(t_range, t_dim, x_range, x_dim):
    # Bounds of 'x' and 't':
    t_min, t_max = t_range
    x_min, x_max = x_range

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T

    return X, t_min, t_max, x_min, x_max, xx, tt, x, t

def heat_1d_boundary_sin_exact(dict_args):
    """
    Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    func_name : str
    """
    t_range = dict_args["t_range"]
    x_range = dict_args["x_range"]
    t_dim = dict_args["t_dim"]
    x_dim = dict_args["x_dim"]
    a = dict_args["a_coeff"]
    n = dict_args["frequency"]

    def heat_eq_exact_solution(x, t, a, L, n):
        """
        Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        return np.exp(-(n**2*np.pi**2*a*t)/(L**2))*np.sin(n*np.pi*x/L)

    X, t_min, t_max, x_min, x_max, xx, tt, x, t = create_t_x_grids(t_range, t_dim, x_range, x_dim)

    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # Obtain the value of the exact solution for each generated point:
    L = x_max - x_min
    y = heat_eq_exact_solution(xx.T, tt.T, a, L, n).T.flatten()[:,None]
    return X, y
