from typing import Optional
import numpy as np
from scipy.linalg import svd


def pca(data_matrix, num_pc: Optional[int] = None):
    """
    This function performs a principal component analysis for a given data matrix.

    https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/
    can be checked for more information

    :param data_matrix:
    :param num_pc: defines the number of principal components that are taken in consideration
    :return:
    """
    # equivalent to L and defines, how much principal components are taken in consideration
    if num_pc is None:
        num_pc = min(data_matrix.shape)
    # centering the matrix
    x_mean = data_matrix - np.mean(data_matrix, axis=0)

    # performing single value decomposition
    u, s, vt = svd(x_mean)

    # create m x n Sigma matrix
    sigma = np.zeros((x_mean.shape[0], x_mean.shape[1]))

    # populate Sigma with m x n diagonal matrix ; x_mean.shape[1]
    sigma[:min(x_mean.shape), :min(x_mean.shape)] = np.diag(s)
    sigma[num_pc:] = 0

    return u, s, vt, sigma


def energy_pca(data_matrix, num_pc: Optional[int] = None):
    """

    :param data_matrix:
    :param num_pc: defines the number of principal components that are taken in consideration
    :return:
    """
    # equivalent to L and defines, how much principal components are taken in consideration
    if num_pc is None:
        num_pc = min(data_matrix.shape)
    # performing pca on data matrix
    _, s, _, sigma = pca(data_matrix, num_pc)
    # calculating trace of squared matrix s
    trace = np.sum(np.square(s))
    # calculating energy matrix
    energy_matrix = np.square(sigma) / trace
    # energies on diagonal of energy matrix
    return energy_matrix.diagonal()
