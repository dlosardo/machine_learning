"""
Abstract class for parametric algorithm
"""
from machine_learning.algorithm.machine_learning_algorithm import (
    MachineLearningAlgorithm)


class ParametricAlgorithm(MachineLearningAlgorithm):
    def __init__(self, cost_function, param_starting_values):
        """
        Parametric Algorithm
        Estimates parameters
        """
        super(ParametricAlgorithm, self).__init__()
        self.cost_function = cost_function
        self.param_starting_values = param_starting_values
        self.nobs = self.cost_function.nobs
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize parameters
        """
        self.cost_function.initialize_parameters(self.param_starting_values)

    def get_parameters(self):
        """
        Return parameters
        """
        return self.cost_function.get_parameters()
