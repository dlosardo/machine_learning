"""
Abstract class for supervised algorithm
"""
from machine_learning.algorithm.machine_learning_algorithm import MachineLearningAlgorithm


class SupervisedAlgorithm(MachineLearningAlgorithm):
    def __init__(self, cost_function, param_starting_values):
        """
        Supervised algorithm.
        Parameters are initialized in this constructor
        :param cost_function A CostFunction object, e.g., SquaredErrorLoss
        :param param_starting_values A dict of the form parameter_name: parameter_value
        """
        super(SupervisedAlgorithm, self).__init__()
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
