import sklearn.metrics

def compute_errors(y_true, y_test):
    return y_test-y_true

def compute_mean_abs_error(y_true, y_predict):
    return sklearn.metrics.mean_absolute_error(y_true, y_predict)

def output_performance(y_true, y_predict):
    print("Mean absolute error: ", compute_mean_abs_error(y_true, y_predict))
    print("Mean squared error: ", sklearn.metrics.mean_squared_error(y_true, y_predict))
    print("Max error: ", sklearn.metrics.max_error(y_true, y_predict))
