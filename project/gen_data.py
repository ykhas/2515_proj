import numpy as np

def gen_data(func_name = "", dict_args = {}):
    """
    Returns the grid X, and solution y, by n data points
    X.shape (n, 2)
    y.shape (n, 1)

    Parameters
    ----------
    func_name : str
    dict_args: dictionary of arguments for the specific heat equation

    """

    X = gen_2d_grid(
        dict_args["x_range"],
        dict_args["t_range"],
        dict_args["x_dim"],
        dict_args["t_dim"])

    return X, gen_solution(func_name, X, dict_args)

def gen_2d_grid(
    x_range: tuple,
    t_range: tuple,
    x_dim: int,
    t_dim: int):
    """
    Returns the grid X with shape (n, 2)

    Parameters
    ----------
    """

    t_min, t_max = t_range
    x_min, x_max = x_range

    # Bounds of 'x' and 't':
    t_min, t_max = t_range
    x_min, x_max = x_range

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    return X

def gen_solution(func_name, X, dict_args):
    """
    Returns solution y by n data points
    y.shape (n, 1)

    Parameters
    ----------
    func_name : str
    X: (n, 2) input data grid of x and t values
    dict_args: dictionary of arguments for the specific heat equation
    """
    if func_name == "":
        return None, None

    return globals()[func_name](X, dict_args)

def heat_1d_boundary_sin_exact(X, dict_args):
    """
    Returns the exact solution for a given x and t (for sinusoidal initial conditions).

    Parameters
    ----------
    func_name : str
    """

    x_range = dict_args["x_range"]
    a = dict_args["a_coeff"]
    n = dict_args["frequency"]
    L = x_range[1] - x_range[0]
    dim = X.shape[0]

    def heat_eq_exact_solution(x, t, a, L, n):
        """
        Returns the exact solution for a given x and t (for sinusoidal initial conditions).

        Parameters
        ----------
        x : np.ndarray
        t : np.ndarray
        """
        return np.exp(-(n**2*np.pi**2*a*t)/(L**2))*np.sin(n*np.pi*x/L)

    y  = np.zeros(dim).reshape(dim)

    # Obtain the value of the exact solution for each generated point:
    for i in range(dim):
        y[i] = heat_eq_exact_solution(X[i, 0],X[i, 1], a, L, n)

    return y
