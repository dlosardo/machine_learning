from numpy import array
from machine_learning.utils.math_utils import get_distance_matrix
from machine_learning.algorithm.nonparametric_algorithm import (
    NonParametricAlgorithm)


class NonParametricDistanceAlgorithm(NonParametricAlgorithm):
    """
    This type of algorithm uses a distance matrix.
    """
    def __init__(self, hypothesis, distance_metric, targets=None):
        """
        """
        super(NonParametricDistanceAlgorithm, self).__init__(hypothesis,
                                                             targets)
        self.distance_metric = distance_metric

    def set_distance_matrix_features(self):
        self.distance_matrix = get_distance_matrix(self.hypothesis.features,
                                                   self.distance_metric)

    def get_distances(self, x):
        """
        Get distances from one observation instance to all observations
        :param x: Numpy array of dimension nfeatures by 1
        :returns: Numpy array of distances from x of length nobs
        """
        distances = array(list(map(
            lambda r: self.distance_metric(x, r), self.hypothesis.features)))
        return distances

    def predict(self, x=None):
        """
        """
        if x is not None:
            distances = map(self.get_distances, x)
            preds = list(map(self.get_predicted_value, distances))
        else:
            self.algorithm()
            preds = [v for k, v in self.predicted_values_map.items()]
        return preds

    def algorithm(self):
        """
        Calculates distances from all features
        and sets the predicted_values_map according to hypothesis
        function for the inherited algorithm type.

        get_predicted_value must take a distance matrix input
        """
        if self.predicted_values_map:
            return
        else:
            distances = map(self.get_distances, self.hypothesis.features)
            preds = list(map(self.get_predicted_value, distances))
            self.predicted_values_map = dict(zip(
                self.targets.flatten(), preds))

    def get_predicted_value(self, distances):
        """
        This needs to be implemented by a child class.
        The child class must implement an algorithm that
        calculates a predicted value as a function of the
        given distances.
        """
        raise NotImplementedError
