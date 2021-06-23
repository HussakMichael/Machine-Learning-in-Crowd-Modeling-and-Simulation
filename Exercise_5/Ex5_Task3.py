import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def finite_diff(X0, X1, delta_t):
    X = (X1 - X0) / delta_t
    return X

def rand_datapoints(x, L):
    idx = np.random.choice(len(x[:,0]), L)
    return x[idx,:]

def rbf(xl, x, epsilon):
    D = euclidean_distances(x, xl, squared=True)
    return np.exp(-np.square(D) / np.square(epsilon))
