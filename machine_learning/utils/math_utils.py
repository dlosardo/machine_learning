"""
Utility functions for mathematical operations
"""
from itertools import combinations
from numpy import (append, ones,
                   diag, sqrt, where, zeros)
from numpy.linalg import norm


def add_constant(numpy_array):
    numpy_array_with_constant = append(ones(numpy_array.shape[0]).reshape(
        numpy_array.shape[0], 1), numpy_array, 1)
    return numpy_array_with_constant


def get_distance_matrix(R, distance_func):
    """
    Computes a symmetric distance matrix
    :param R: A numpy array of features of dimension
     nfeatures x nobs
    :param distance_func: A function that calculates the
     distance between two vectors
    :returns A symmetric nfeatures x nfeatures matrix
    """
    nrows = R.shape[0]
    distance_matrix = zeros(nrows * nrows).reshape(nrows, nrows)
    for c in combinations(range(0, len(R)), 2):
        distance_matrix[c] = distance_func(R[c[0]], R[c[1]])
    distance_matrix = distance_matrix.T + (
        distance_matrix - diag(diag(distance_matrix)))
    return distance_matrix


"""
Distance Functions
"""


def euclidean(v1, v2):
    return sqrt(norm(v1)**2 + norm(v2)**2 - 2*v1.dot(v2))


def jaccard(v1, v2):
    """
    Distance metric
    1 - (number same items) / (items in either)
    """
    v1_indices = where(v1 == 1)[0]
    v2_indices = where(v2 == 1)[0]
    nboth = len(set(v1_indices).intersection(v2_indices))
    nneither = len(set(v1_indices).union(v2_indices))
    return 1 - (nboth)/(nneither)
