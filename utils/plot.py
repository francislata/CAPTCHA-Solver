import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt

def plot(x, y, x_title, y_title, title, filepath=None):
    plt.clf()

    plt.subplot(111)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    if filepath:
        plt.savefig(filepath)
