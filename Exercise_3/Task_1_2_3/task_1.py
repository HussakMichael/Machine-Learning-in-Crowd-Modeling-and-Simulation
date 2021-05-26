# -*- coding: utf-8 -*-

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig


def phase_portrait(matrix: np.ndarray, start: list,
                   x_dimension: tuple = (-1, 1), y_dimension: tuple = (-1, 1),
                   sections: int = 10, name: str=""):
    """
    This function plots a phase portrait

    inspired by https://pysd-cookbook.readthedocs.io/en/latest/analyses/visualization/phase_portraits.html

    :param matrix: specifies the used matrix
    :param start: list of initial points
    :param x_dimension: defines the dimensions in x direction
    :param y_dimension: defines the dimensions in y direction
    :param sections: defines the amount of sections, the mgrid is divided into
    :return: None
    """
    # calculating eigenvalues of matrix
    eigenvalues = eig(matrix)[0]
    # creating mgrid
    y, x = np.mgrid[y_dimension[0]:y_dimension[1]:complex(sections),
                    x_dimension[0]:x_dimension[1]:complex(sections)]

    u = matrix[0][0] * x + matrix[0][1] * y
    v = matrix[1][0] * x + matrix[1][1] * y

    # plotting phase portrait
    plt.figure(figsize=(6, 6))
    plt.streamplot(x, y, u, v, start_points=start, density=10)
    plt.quiver(x, y, u, v)
    plt.axis('scaled')
    plt.axis([-1, 1, -1, 1])
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(fr'$\alpha={matrix[0][0]}, \lambda_1={eigenvalues[0]}, \lambda_2={eigenvalues[1]}$')
    plt.savefig(f"phase_portrait_{name}.png")
    plt.show()


def draw_eigenvalues(matrix: np.ndarray, name: str=""):
    """
    This function plots the eigenvalues of a given matrix

    :param matrix: specifies the used matrix
    :return: None
    """
    # calculating eigenvalues
    eigenvalues = eig(matrix)[0]
    plt.figure(figsize=(10, 10))
    # plotting eigenvalues, real and imaginative
    plt.scatter(eigenvalues.real, eigenvalues.imag)
    plt.axis('square')
    # centering plot
    yabs_max = abs(max(plt.gca().get_ylim(), key=abs))
    xabs_max = abs(max(plt.gca().get_xlim(), key=abs))
    plt.gca().set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.gca().set_xlim(xmin=-xabs_max, xmax=xabs_max)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    plt.gca().spines['left'].set_position('center')
    plt.gca().spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    # adding title, axis labels and grid
    plt.title(fr'Eigenvalues $\lambda_1={eigenvalues[0]}$ and $\lambda_2={eigenvalues[1]}$')
    plt.gca().set_ylabel('Imaginary', loc='top')
    plt.gca().set_xlabel('Real', loc='right')
    plt.grid(True)
    plt.savefig(f"eigenvalue_plot_{name}.png")
    plt.show()
