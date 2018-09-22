"""
Abstract class for non-parametric algorithms
"""
from machine_learning.algorithm.machine_learning_algorithm import (
    MachineLearningAlgorithm)


class NonParametricAlgorithm(MachineLearningAlgorithm):
    def __init__(self, hypothesis, targets=None):
        """
        Non Parametric Algorithm
        Does not estimate finite parameters but uses the data
        to obtain estimates. Thus, may take targets.
        """
        super(NonParametricAlgorithm, self).__init__()
        self.hypothesis = hypothesis
        self.nfeatures = self.hypothesis.nfeatures
        self.targets = targets
        self.nobs = self.hypothesis.nobs
        self.predicted_values_map = {}
