import numpy as np
import matplotlib.pyplot as plt


def _equation(alpha: float, a: float, b: float) -> tuple:
    """
    Using a general form of equation: x_dot = alpha - a*x^2 - b
    as it can be applied on both equations requested in the exercise

    :param alpha: specifies the value of alpha
    :param a: specifies the parameter a in the above mentioned equation
    :param b: specifies the parameter b in the above mentioned equation
    :return: tuple consisting of (bifur_starting_alpha, pos_attr_alpha, neg_attr_alpha)
    """
    alpha_values = np.where(alpha < b, np.nan, alpha)
    # equation solved to x
    x = np.sqrt((alpha_values - b)/a)
    return b, x, -x


def bifurcation(alpha: np.array, a: float, b: float, name=""):
    """
    This function is creating a bifurcation plot

    Using a general form of equation: x_dot = alpha - a*x^2 - b
    as it can be applied on both equations requested in the exercise

    :param alpha: specifies the value of alpha
    :param a: specifies the parameter a in the above mentioned equation
    :param b: specifies the parameter b in the above mentioned equation
    :return: None
    """
    bifur_starting_alpha, pos_attr_alpha, neg_attr_alpha = _equation(alpha, a, b)

    plt.plot(alpha, pos_attr_alpha, '-', label='stable')
    plt.plot(alpha, neg_attr_alpha, '-', label='unstable')
    plt.plot(bifur_starting_alpha, 0, 'rx', label='bifurcation start')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('x')
    plt.xlim(-2, 8)
    plt.title(fr"Bifurcation plot of $\.x = \alpha - {a} x^{2} - {b}$ with $\alpha$ from {min(alpha)} to {max(alpha)}")
    plt.legend()
    plt.savefig(f"task_2_bifurcation_plot_{name}.png")
    plt.show()


def phase_portrait(equation_parameters: tuple,
                   x_dimension: tuple = (-2, 2), y_dimension: tuple = (-1, 1),
                   sections: int = 10, name=""):
    """
    This function plots a phase portrait

    inspired by https://pysd-cookbook.readthedocs.io/en/latest/analyses/visualization/phase_portraits.html

    :param equation_parameters: the parameters of the equation in the form (alpha, a, b)
    :param x_dimension: defines the dimensions in x direction
    :param y_dimension: defines the dimensions in y direction
    :param sections: defines the amount of sections, the mgrid is divided into
    :return: None
    """
    # creating mgrid
    x_2, x_1 = np.mgrid[y_dimension[0]:y_dimension[1]:complex(sections),
                        x_dimension[0]:x_dimension[1]:complex(sections)]
    # unpacking equation_parameters
    alpha, a, b = equation_parameters
    u = alpha - a * (x_1 ** 2) - b
    # x_2 does not contribute in bifurcation (its dynamic is v = -x_2)
    v = -x_2

    # plotting phase portrait
    plt.figure(figsize=(6, 6))
    plt.streamplot(x_1, x_2, u, v)
    plt.quiver(x_1, x_2, u, v)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(fr"$\.x=\alpha - {a}x^{2} - {b}$ with $\alpha={alpha}$")
    plt.savefig(f"task_2_phase_portrait_{name}.png")
    plt.show()
