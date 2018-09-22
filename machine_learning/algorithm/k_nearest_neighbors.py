from numpy import argpartition
from machine_learning.algorithm.nonparametric_distance_algorithm import (
    NonParametricDistanceAlgorithm)


class KNearestNeighbors(NonParametricDistanceAlgorithm):
    def __init__(self, hypothesis, targets, distance_metric):
        super(KNearestNeighbors, self).__init__(hypothesis,
                                                distance_metric,
                                                targets)

    def get_predicted_value(self, distances):
        """
        Given distances, calculates k nearest neighbors
        """
        top_indices_to_k = argpartition(distances.flatten(),
                                        self.hypothesis.k)[:self.hypothesis.k]
        sorted_indices_distances = sorted(zip(
            top_indices_to_k, distances[
                top_indices_to_k].flatten()),
            key=lambda x: x[1])
        sorted_indices_to_k, sorted_distances_to_k = map(
            list, zip(*sorted_indices_distances))
        k_nearest_neighbors = self.targets[sorted_indices_to_k]
        return self.hypothesis.hypothesis_function(
            k_nearest_neighbors, sorted_distances_to_k)[0]
