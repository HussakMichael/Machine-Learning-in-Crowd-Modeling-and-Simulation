import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def phase_portrait(alpha, x_dimension=(-2, 2), y_dimension=(-2, 2), sections=10):
    """
    This function plots a phase portrait

    inspired by https://pysd-cookbook.readthedocs.io/en/latest/analyses/visualization/phase_portraits.html

    :param alpha: defines the value alpha
    :param x_dimension: defines the dimensions in x direction
    :param y_dimension: defines the dimensions in y direction
    :param sections: defines the amount of sections, the mgrid is divided into
    :return: None
    """

    # creating mgrid
    x_2, x_1 = np.mgrid[y_dimension[0]:y_dimension[1]:complex(sections),
                        x_dimension[0]:x_dimension[1]:complex(sections)]

    # Andronov-Hopf bifurcation
    u = alpha * x_1 - x_2 - x_1 * (x_1 ** 2 + x_2 ** 2)
    v = x_1 + alpha * x_2 - x_2 * (x_1 ** 2 + x_2 ** 2)

    # plotting phase portrait
    plt.figure(figsize=(6, 6))
    plt.streamplot(x_1, x_2, u, v)
    plt.quiver(x_1, x_2, u, v)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis('scaled')
    plt.grid(True)
    plt.title(fr"Andronov Hopf bifurcation for $\alpha={alpha}$")
    plt.savefig(f"phase_portrait_andronov_hopf_alpha_{alpha}.png")
    plt.show()


def andronov_hopf_equation(t, y):
    """
    This function is used by the solve_ivp function.
    It returns the vector field of the Andronov-Hopf bifurcation

    :param t: time variable, necessary for the use in solve_ivp
    :param y: variable y
    :return: vector field of andronov hopf bifurcation
    """
    alpha = 1
    x1, x2 = y
    u = alpha * x1 - x2 - x1 * (np.square(x1) + np.square(x2))
    v = x1 + alpha * x2 - x2 * (np.square(x1) + np.square(x2))
    return u, v


def orbit_plot(starting_point: np.array, t_start: int = 0, t_stop: int = 15, step: int = 150):
    """
    This function is used to plot the orbit of the andronov hopf equation

    :param starting_point: Defines the starting point
    :param t_start: defines the starting time
    :param t_stop: defines the stop time
    :param step: defines the amount of steps
    :return:
    """
    # solving the andronov hopf equation
    sol = solve_ivp(
        andronov_hopf_equation, t_span=(t_start, t_stop), y0=starting_point,
        t_eval=np.linspace(t_start, t_stop, step))

    # plotting orbit
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.t, label=f'Andronov Hopf for starting point {starting_point}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('time')
    ax.legend()
    plt.savefig(f"task_3_orbit_plot_{starting_point}.png")
    plt.show()


def cusp_bifurcation_surface(view_angle):
    """
    This function is used to plot the surface of the cusp bifurcation

    :return:
    """
    # iterating over alpha_1, alpha_2 and the solutions of the cusp equation
    xs, ys, zs = [], [], []
    for alpha_1 in np.linspace(-10, 10, 50):
        for alpha_2 in np.linspace(-10, 10, 50):
            sol = np.roots([-1, 0, alpha_2, alpha_1])
            for z in sol:
                # only extracting the real values
                if np.isreal(z):
                    xs.append(alpha_1)
                    ys.append(alpha_2)
                    zs.append(np.real(z))

    # plotting cusp bifurcation surface
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')
    ax.set_zlabel('x')
    ax.set_title('Cusp bifurcation surface')
    ax.scatter(xs, ys, zs)
    ax.view_init(*view_angle)
    plt.savefig("task_3_cusp_bifurcation_surface.png")
    plt.show()
