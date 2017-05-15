from numpy import dot
from numpy.linalg import norm
from machine_learning.cost_function.perceptron_cost import PerceptronCost
from machine_learning.learning_type import LearningTypes


class PerceptronBatchCost(PerceptronCost):
    """Batch Learning (consider all data at once)
    """
    def __init__(self, hypothesis, targets):
        super(PerceptronBatchCost, self).__init__(hypothesis, targets)
        self.learning_type = LearningTypes.BATCH

    def cost_function(self):
        error_matrix = self.get_error_matrix()
        costs = ((error_matrix.T).dot(self.hypothesis.features))/self.nobs
        cost_length = norm(costs)
        return cost_length

    def cost_function_derivative(self):
        error_matrix = self.get_error_matrix()
        return -1. * (1./(2.*self.nobs))*(error_matrix.T).dot(self.hypothesis.features).T

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        print(new_cost)
        return new_cost <= tolerance
