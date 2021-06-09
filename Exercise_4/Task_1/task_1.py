# General imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.misc
from skimage.transform import resize
# Local imports
from pca import energy_pca, pca
# Typing imports
from typing import Union


def draw_pc(df: pd.DataFrame, vt, s):
    """
    This function is used to draw the principal components and plot the data points

    :param df: dataframe containing the data points
    :param vt: matrix vt
    :param s: singular values
    :return: None
    """
    # calculating the mean of the data points --> needed for start point
    x_mean, f_x_mean = df.mean()

    fig = plt.figure()
    ax = fig.add_subplot()
    # plotting data points
    ax.scatter(df["x"], df["f(x)"])
    # plotting principal components
    for i, principal_comp in enumerate(vt):
        ax.quiver(x_mean, f_x_mean, principal_comp[0], principal_comp[1], label=f"Principal Component {i+1}",
                  color=f"C{i+1}", scale_units='xy', scale=1)

    plt.title(f"Principal Component 1 with singular value {np.around(s[0], 2)} "
              f"and Principal Component 2 with singular values: {np.around(s[1],2)}")
    plt.legend(loc="best")
    plt.savefig("task1_1_principal_components.png", bbox_inches='tight')
    plt.show()


def draw_racoon_various_l(num_pc: Union[None, int]):
    """
    This function draws the reconstructed image of the racoon depending on the specified principal components

    :param num_pc: specifies the number of principal components, if None, all principal components are used
    :return: None
    """
    # preprocessing
    face = scipy.misc.face(gray=True)
    face = resize(face, (185, 249))
    # performing pca
    u, s, vt, sigma = pca(face, num_pc)
    # reconstructing image
    reconstruction = u @ sigma @ vt
    # summing up energies
    energy = np.sum(energy_pca(face, num_pc))
    # energy loss in percent
    energy_loss = (1 - energy) * 100
    # creating image
    fig = plt.figure()
    plt.imshow(reconstruction, cmap=plt.cm.gray)
    plt.title(fr"Reconstructed Racoon with {num_pc if num_pc is not None else 'all'} "
              fr"Principal components (E: {np.round(energy, 3)}, $\Delta$E: {np.round(energy_loss, 3)}%)")
    plt.axis('off')
    plt.savefig(f"task1_2_racoon_l_{num_pc if num_pc is not None else 'all'}.png", bbox_inches='tight')
    plt.show()


def find_1_percent_loss_l():
    """
    This function finds the number of principal components, where the energy loss is greater than 1%
    and draws the reconstructed image of the racoon

    :return:
    """
    energy = 1.0
    # preprocessing
    face = scipy.misc.face(gray=True)
    face = resize(face, (185, 249))
    # need to be less than 120, as there the energy loss is at 0.2
    num_pc = 120
    while energy > 0.99:
        # summing up energies
        energy = np.sum(energy_pca(face, num_pc))
        # updating number of principal components
        num_pc -= 1
    return draw_racoon_various_l(num_pc)


def draw_pedestrians_path(df: pd.DataFrame, pedestrians: list):
    """
    This function draws the path of the specified pedestrians

    :param df: The trajectories of the pedestrians in the form [x_1, y_1, x_2, y_2, ...]
    :param pedestrians: list of pedestrians to be drawn, e.g. [1, 2]
    :return: None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for ped in pedestrians:
        ax.scatter(df[f"x_{ped}"], df[f"y_{ped}"], label=f"Pedestrian {ped}", c=f"C{ped-1}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Path of Pedestrians {*pedestrians,}")
    ax.legend()
    plt.savefig("task1_3_ped_path.png", bbox_inches='tight')
    plt.show()


def project_on_x_principal_components(data_matrix: np.array, num_pc: None):
    """
    This function draws the projection on the defined principal components

    https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd
    https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

    :param data_matrix: specifies the data points in a matrix
    :param num_pc: defines the number of principal components that are taken in consideration, if None, all are taken
    :return: None
    """
    # performing pca
    u, s, vt, sigma = pca(data_matrix, num_pc)

    reconstruction = u @ sigma
    # summing up energies
    energy = np.sum(energy_pca(data_matrix, num_pc))

    # creating plot
    fig = plt.figure(figsize=(10, 10))
    if num_pc == 2:
        ax = fig.add_subplot()
        ax.scatter(reconstruction[:, 0], reconstruction[:, 1], color="C0")
        ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2")
    elif num_pc == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(reconstruction[:, 0], reconstruction[:, 1], reconstruction[:, 2], color="C0")
        ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2", zlabel="Principal Component 3")
    else:
        raise Exception(f"Function project_on_x_principal_components only able to draw projection for 3 PCs")
    ax.set_title(f"Projection on first {num_pc} Principal Components (E: {np.round(energy * 100, 3)}%)")
    plt.savefig(f"task1_3_l_{num_pc}_principal_components.png", bbox_inches='tight')
    plt.show()


def draw_trajectory_x_principal_components(data_matrix: np.array, num_pc: None):
    """
    This function draws the trajectory of the first 2 pedestrians

    https://stackoverflow.com/questions/60508233/python-implement-a-pca-using-svd
    https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca

    :param data_matrix: specifies the data points in a matrix
    :param num_pc: defines the number of principal components that are taken in consideration, if None, all are taken
    :return: None
    """
    # performing pca
    u, s, vt, sigma = pca(data_matrix, num_pc)

    reconstruction = u @ sigma @ vt
    # summing up energies
    energy = np.sum(energy_pca(data_matrix, num_pc))
    # creating plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for ped in range(2):
        ax.scatter(reconstruction[:, ped*2], reconstruction[:, ped*2+1],
                   label=f"Trajectory Ped {ped+1}", color=f"C{ped}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(
        f"Trajectory of the first two Pedestrian projected on {num_pc} Principal Components "
        f"(E: {np.round(energy * 100, 3)}%)")
    ax.legend()
    plt.savefig(f"task1_3_l_{num_pc}_trajectory.png", bbox_inches='tight')
    plt.show()


def plot_energy_over_pc(data_matrix: np.array):
    """
    This function plots the energy over the used principal components

    :param data_matrix: specifies the data points in a matrix
    :return:
    """
    pcs = []
    energy = []
    for pc in range(min(data_matrix.shape)):
        pcs.append(pc)
        energy.append(sum(energy_pca(data_matrix, pc)))
    plt.figure(figsize=(10, 10))
    plt.plot(pcs, energy)
    plt.xlabel('number of components')
    plt.ylabel('Energy')
    plt.savefig("task1_2_energy_over_pc.png", bbox_inches='tight')
    plt.show()
