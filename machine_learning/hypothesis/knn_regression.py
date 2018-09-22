"""
K-Nearest Neighbors Regression
Y_hat = sum(weights * Y) / sum(weights)
        for Ys corresponding to Xs in neighborhood of size k
Default for weights is a vector of ones (unweighted mean)
"""
from numpy import ones, array
from machine_learning.hypothesis.nonparametric_hypothesis import (
    NonParametricHypothesis)


class KNearestNeighborsRegression(NonParametricHypothesis):
    def __init__(self, features, k, weights=None):
        """
        """
        super(KNearestNeighborsRegression, self).__init__(features)
        self.k = k
        self.weights = weights

    def hypothesis_function(self, k_nearest_neighbors,
                            distances):
        if self.weights is None:
            weights = ones(len(distances))
        elif self.weights == "inverse_distance":
            weights = array([1. / d
                             if d != 0 else 0
                             for d in distances])
        nearest_neighbor = weights.dot(
                k_nearest_neighbors) / sum(weights)
        return nearest_neighbor
