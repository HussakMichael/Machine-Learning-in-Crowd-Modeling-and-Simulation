import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def linear_approximation(df: pd.DataFrame, rcond=None, name: str = ""):
    """
    This function can be used to linearly approximate a given dataset.
    The linear approximation and the data is plotted.

    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    :param df: Pandas DataFrame with Column x and Column f(x)
    :param rcond: Cut-off ratio for small singular values of a in np.linalg.lstsq
    :param name: name of the used dataset
    :return:
    """
    A = np.linalg.lstsq(df.iloc[:, :1], df.iloc[:, 1:], rcond=rcond)[0]
    print(f"Matrix A: {A}")
    xi = np.linspace(df['x'].min(), df['x'].max(), len(df)).reshape(-1, 1)
    fx_predicted = xi @ A
    """
    Alternative to above
    # rewriting the line equation as f(x) = Ap, where A = [[x 1]] and p = [[m], [c]]
    A_ = np.vstack([df["x"], np.ones(len(df["x"]))]).T
    # use lstsq to solve for p
    m, c = np.linalg.lstsq(A_, df["f(x)"], rcond=rcond)[0]
    print(f"m: {m} and c: {c}")
    """
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot()
    # plotting data along with fitted line
    ax.scatter(df["x"], df["f(x)"], label="Original Data")
    # plt.plot(df["x"], m * df["x"] + c, 'r', label='Fitted line')
    plt.plot(xi, fx_predicted, 'r', label='Fitted line')
    plt.title(f"Visualization of {name}.txt")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.savefig(f"task_1_lin_approx_{name}.png")
    plt.show()


def rbf(distances, epsilon):
    return np.exp(-np.square(distances) / np.square(epsilon))


def least_squares_approx(df, x_l, e, rcond=None, return_res=False):
    # distances = cdist(df.iloc[:, :1], df.iloc[:, 1:].sample(n=L))
    distances = cdist(df.iloc[:, :1], x_l)
    epsilon = e * np.max(distances)
    phi = rbf(distances, epsilon)

    if return_res:
        X, residuals = np.linalg.lstsq(phi, df.iloc[:, 1:], rcond=rcond)[:2]
        return X, epsilon, residuals

    X = np.linalg.lstsq(phi, df.iloc[:, 1:], rcond=rcond)[0]
    return X, epsilon


def plot_approx(df, xi, fx_predicted, L, epsilon, name=""):
    fig = plt.figure(figsize=(10, 10))
    plt.plot(xi, fx_predicted, color='r', label='Approx Func')
    plt.scatter(df.iloc[:, :1], df.iloc[:, 1:], label='Data Points')
    plt.title(fr"Approximation plot for L={L} and $\epsilon$={epsilon}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.savefig(f"task_1_rbf_approx_{name}_l_{L}_eps_{epsilon}.png")
    plt.show()
