"""
Gradient Descent ML Algorithm
IS an IterativeSupervisedAlgorithm
"""
from machine_learning.algorithm.iterative_supervised_algorithm import (
    IterativeSupervisedAlgorithm)


class GradientDescent(IterativeSupervisedAlgorithm):
    def __init__(self, cost_function, learning_rate=None, tolerance=None,
                 param_starting_values=None):
        """
        Constructor for GradientDescent
        :param learning_rate: A float value representing the learning rate
        :param tolerance: A float value representing the tolerance value
            used to inform convergence specifications
        """
        super(GradientDescent, self).__init__(cost_function,
                                              param_starting_values)
        if learning_rate is None:
            self.learning_rate = 1.
        else:
            self.learning_rate = learning_rate
        if tolerance is None:
            self.tolerance = .001
        else:
            self.tolerance = tolerance
        self.current_cost = None
        self.new_cost = None

    def reset(self):
        self.current_cost = None
        self.new_cost = None
        self.iter = 0
        self.converged = False
        self.cost_function_list = []
        self.convergence_value_list = []
        self.initialize_parameters()

    def convergence_criteria_met(self):
        return self.cost_function.convergence_criteria_met(
            self.current_cost, self.new_cost, self.tolerance)

    def convergence_value(self):
        return self.cost_function.convergence_value(
            self.current_cost, self.new_cost)
