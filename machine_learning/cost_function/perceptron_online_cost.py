"""
cost is number of errors made calculated as:
J(weights) = targets - hypothesis(weights)
number_errors = sum(abs(J(weights)))
cost is defined as:
derivative of weights wrt parameters:
(targets(i) - hypothesis(i)) %*% features(i)
note that this is calculated across the observations one by one
and each time the weights are updated. This means that ordering
the inputs differently may lead to different results.
"""

from numpy import dot, sum, abs
from machine_learning.cost_function.perceptron_cost import PerceptronCost
from machine_learning.model_utils.learning_type import LearningTypes


class PerceptronOnlineCost(PerceptronCost):
    """Online Learning (consider each data point as it comes in)
    """
    def __init__(self, hypothesis, targets):
        super(PerceptronOnlineCost, self).__init__(hypothesis, targets)
        self.learning_type = LearningTypes.ONLINE

    def cost_function(self):
        """Computes the perceptron loss cost function
        :returns a 1 x 1 np array containing a float value representing the
         value of the cost function, here the number of errors made
        """
        error_matrix = self.get_error_matrix()
        number_errors = sum(abs(error_matrix))
        return number_errors

    def cost_function_derivative(self, index):
        # The partial derivative of the cost function with respect to the parameters
        y = self.hypothesis.hypothesis_function(index)
        return (self.targets[index] - y).dot(
                self.hypothesis.features[index, :].reshape(1, self.hypothesis.nparams)).reshape(
                        self.hypothesis.nparams, 1)

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        return self.convergence_value(current_cost, new_cost) == 0

    def convergence_value(self, current_cost, new_cost):
        return new_cost
