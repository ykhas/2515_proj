from dataclasses import dataclass

import numpy as np
import deepxde as dde

from project.gen_data import create_xt_grids, gen_solution
from project.post_processing import Plotter
from project.timer import Timer
from project.metric import *

TIMER_REPEAT_TIMES = 25

# Problem parameters:
@dataclass
class PinnConstParam:
    a = 0.4 # Thermal diffusivity
    L = 1 # Length of the bar, Assum x starts at 0
    n = 1 # Frequency of the sinusoidal initial conditions
    t_end = 1 # Assume t_start  = 0
    test_x_dim = 60
    test_t_dim = 200
    analytical_function_name = "heat_1d_boundary_sin_exact"
    numerical_function_name = "finite_difference_crank_nicolson"
    X_test = None
    y_analytical = None
    y_numerical = None

    def __str__(self) -> str:
        rs = "Pinn Const Param: \n"
        rs =  rs + "a: {}\n".format(self.a)
        rs =  rs + "L: {}\n".format(self.L)
        rs =  rs + "n: {}\n".format(self.n)
        rs =  rs + "t_end: {}\n".format(self.t_end)
        rs =  rs + "test_x_dim: {}\n".format(self.test_x_dim)
        rs =  rs + "test_t_dim: {}\n".format(self.test_t_dim)
        rs =  rs + "analytical_function_name: " + self.analytical_function_name + "\n"
        rs =  rs + "numerical_function_name: " + self.numerical_function_name + "\n"
        return rs

@dataclass
class PinnTestParam:
    d_num_domain = 2540
    d_num_boundary = 80
    d_num_initial = 160
    nn_hidden_layer_size = 20
    nn_hidden_layer_num = 3

    def __str__(self) -> str:
        rs = "Pinn Test Param: \n"
        rs =  rs + "d_num_domain: {}\n".format(self.d_num_domain)
        rs =  rs + "d_num_boundary: {}\n".format(self.d_num_boundary)
        rs =  rs + "d_num_initial: {}\n".format(self.d_num_initial)
        rs =  rs + "nn_hidden_layer_size: {}\n".format(self.nn_hidden_layer_size)
        rs =  rs + "nn_hidden_layer_num: {}\n".format(self.nn_hidden_layer_num)
        return rs

def solve_analytical_and_numerical(const_params:PinnConstParam):
    const_params.X_test, x, t = create_xt_grids(
        x_range= (0, const_params.L),
        x_dim = const_params.test_x_dim,
        t_range= (0, const_params.t_end),
        t_dim = const_params.test_t_dim)

    ti = Timer(TIMER_REPEAT_TIMES)
    const_params.y_analytical = ti.time_average(lambda: gen_solution(
        const_params.X_test, x, t,
        const_params.analytical_function_name,
        {
            "a_coeff": const_params.a,
            "frequency": const_params.n,
        }
    ))
    print(const_params.analytical_function_name + ": "+ ti.str_average())

    const_params.y_numerical = ti.time_average(lambda : gen_solution(
        const_params.X_test, x, t,
        const_params.numerical_function_name,
        {
            "a_coeff": const_params.a,
        }))
    print(const_params.numerical_function_name + ": "+ ti.str_average())

def pde(x, y, a):
    """
    Expresses the PDE residual of the heat equation.
    """
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a*dy_xx

def create_model(const_params: PinnConstParam, test_params: PinnTestParam):
    # Computational geometry:
    geom = dde.geometry.Interval(0, const_params.L)
    timedomain = dde.geometry.TimeDomain(0, const_params.t_end)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Initial and boundary conditions:
    bc = dde.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
    ic = dde.IC(
        geomtime, lambda x: np.sin(const_params.n*np.pi*x[:, 0:1]/const_params.L) , lambda _, on_initial: on_initial
        )

    l_pde = lambda x, y: pde(x, y, const_params.a)

    geomtime_data = dde.data.TimePDE(
        geomtime, l_pde, [bc, ic],
        num_domain=test_params.d_num_domain,
        num_boundary=test_params.d_num_boundary,
        num_initial=test_params.d_num_initial,
        num_test=2540
    )

    nn_architecture = [2] + [test_params.nn_hidden_layer_size] * test_params.nn_hidden_layer_num + [1]
    net = dde.nn.FNN(
        nn_architecture,
        "tanh",
        "Glorot normal")

    model = dde.Model(geomtime_data, net)

    return geomtime_data, model

def train_model(model):
    # Build and train the model:
    ti = Timer()
    ti.start()
    model.compile("adam", lr=1e-3)
    model.train(epochs=20000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    ti.stop()
    print("Training time: "+ ti.str_elapsed_time())

    return losshistory, train_state

def plot_train_data(geomtime_data):
    X_bc_train = geomtime_data.bc_points()
    print("train_bc_points shape", X_bc_train.shape)

    plotter = Plotter(1, 2)
    plotter.plot_2d(X_bc_train, "train_bc_points")

    # Train points are not used anymore because deepxde train_points(),
    # always generate random train points, but it is not what we are doing to
    # train the nn, in train_next_batch(). train_next_batch, isnt random!!
    # X_train_points = geomtime_data.train_points()
    # print("train_points shape", X_train_points.shape)
    # plotter.plot_2d(X_train_points, "train_domain_points")

    X_train_next_batch, _, _ = geomtime_data.train_next_batch()
    print("train_next_batch shape", X_train_next_batch.shape)
    plotter.plot_2d(X_train_next_batch, "train_next_batch")

def plot_analytical_and_numerical(plotter, const_params: PinnConstParam):
    # I. Analytical test solution
        plotter.plot_2d_y(
            const_params.X_test,
            const_params.y_analytical,
            const_params.analytical_function_name)
        plotter.plot_2d_colormesh(
            const_params.y_analytical,
            const_params.test_x_dim,
            const_params.test_t_dim)

    # II. Numerical test solution
        plotter.plot_2d_y(
            const_params.X_test,
            const_params.y_numerical,
            const_params.numerical_function_name)
        plotter.plot_2d_colormesh(
            const_params.y_numerical,
            const_params.test_x_dim,
            const_params.test_t_dim)

def predict_and_output_report(const_params: PinnConstParam, model,
    losshistory = None,
    train_state = None,
    test_params: PinnTestParam = None):
    plotter = Plotter(4, 2)

    plot_analytical_and_numerical(plotter, const_params)

    # III. Physics Informed Neural Network test solution``
    function_name = "PINN prediction"
    ti = Timer(TIMER_REPEAT_TIMES)
    y_pred = ti.time_average(lambda : model.predict(const_params.X_test))
    print(function_name + ": "+ ti.str_average())
    plotter.plot_2d_y(const_params.X_test, y_pred, function_name)
    plotter.plot_2d_colormesh(
        y_pred,
        const_params.test_x_dim,
        const_params.test_t_dim)

    # IV. Performance metrics
    errors = compute_errors(const_params.y_analytical, y_pred)
    plotter.plot_2d_y(const_params.X_test, errors, "PINN vs Analytical errors")
    plotter.plot_2d_colormesh(
        errors,
        const_params.test_x_dim,
        const_params.test_t_dim)

    if losshistory and train_state:
        dde.saveplot(losshistory, train_state, issave=False, isplot=True)
        print("IGNORE ^^^ PDE test data SOLUTION PLOT!!! ^^^")

    f = model.predict(const_params.X_test, operator=pde)
    print("---Performance measured wrt analytical solution---")
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(const_params.y_analytical, y_pred))
    output_performance(const_params.y_analytical, y_pred)
    # np.savetxt("test.dat", np.hstack((X_test, y_analytical, y_pred)))

    print(const_params)
    if test_params != None:
        print(test_params)
