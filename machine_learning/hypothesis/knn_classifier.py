"""
K-Nearest Neighbors Classifier
"""
import random
from numpy import flatnonzero
from scipy.stats import itemfreq
from machine_learning.hypothesis.nonparametric_hypothesis import (
    NonParametricHypothesis)


class KNearestNeighborsClassifier(NonParametricHypothesis):
    def __init__(self, features, k):
        """
        """
        super(KNearestNeighborsClassifier, self).__init__(features)
        self.k = k

    def hypothesis_function(self, k_nearest_neighbors,
                            distances):
        freqs = itemfreq(k_nearest_neighbors)
        counts = freqs[:, 1]
        nearest_neighbor_index = random.choice(
            flatnonzero(counts == counts.max()))
        nn = freqs[:, 0][nearest_neighbor_index]
        return nn
