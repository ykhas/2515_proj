import matplotlib.pyplot as plt

def plot_2d_y(X, y, label = ""):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y)
    ax.title.set_text(label)

def plot_2d(X, label = ""):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1])
    ax.title.set_text(label)
