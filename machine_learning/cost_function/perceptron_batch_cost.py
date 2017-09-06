"""
cost is ERROR_MATRIX * FEATURES / nobs
where ERROR_MATRIX is calculated as:
ERROR_MATRIX = targets - hypothesis(weights)
and the length of the costs matrix is the actual cost
"""
from numpy.linalg import norm
from numpy import array
from machine_learning.cost_function.perceptron_cost import PerceptronCost
from machine_learning.model_utils.learning_type import LearningTypes


class PerceptronBatchCost(PerceptronCost):
    def __init__(self, hypothesis, targets):
        """
        Batch Learning (consider all data at once)
        """
        super(PerceptronBatchCost, self).__init__(hypothesis, targets)
        self.learning_type = LearningTypes.BATCH

    def cost_function(self):
        error_matrix = self.get_error_matrix()
        costs = ((error_matrix.T).dot(self.hypothesis.features))/self.nobs
        cost_length = array([norm(costs)]).reshape(1, 1)
        return cost_length

    def cost_function_derivative(self):
        error_matrix = self.get_error_matrix()
        return -1. * (1./(2.*self.nobs))*(error_matrix.T).dot(self.hypothesis.features).T

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        return self.convergence_value(current_cost[0], new_cost[0]) <= tolerance

    def convergence_value(self, current_cost, new_cost):
        return new_cost[0]
