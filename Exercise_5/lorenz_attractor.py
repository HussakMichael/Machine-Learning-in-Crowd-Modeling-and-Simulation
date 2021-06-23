import numpy as np

def lorenz(x, y, z, s, r, b):
    """
    This function computes the lorenz attractor
    :param x: x-coordinate
    :param y: y-coordinate
    :param z: z-coordinate
    :param s: sigma parameter
    :param r: rho parameter
    :param b: beta parameter
    :return: x_dot, y_dot, z_dot
    """
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def time_delay_lorenz(x, dn):
    """
    This function computes the time delayed embedding of the lorenz attractor
    :param x: data
    :param dn: time-delay
    """
    x_t = x
    x_t_delta_t = np.roll(x, dn)
    x_t_delta_2t = np.roll(x, 2*dn)
    return x_t, x_t_delta_t, x_t_delta_2t
