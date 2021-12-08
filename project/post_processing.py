import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, rows, cols):
        self.fig = plt.figure()
        standard_size_in = 5
        self.fig.set_figheight(standard_size_in * rows)
        self.fig.set_figwidth(standard_size_in * cols)
        self.rows = rows
        self.cols = cols
        self.idx = 1

    def plot_2d_y(self, X, y, label = ""):
        ax = self.fig.add_subplot(self.rows, self.cols, self.idx, projection='3d')
        self.idx = self.idx + 1
        ax.scatter(X[:, 0], X[:, 1], y)
        ax.title.set_text(label)

    def plot_2d(self, X, label = ""):
        ax = self.fig.add_subplot(self.rows, self.cols, self.idx, projection='3d')
        self.idx = self.idx + 1
        ax.scatter(X[:, 0], X[:, 1])
        ax.title.set_text(label)
