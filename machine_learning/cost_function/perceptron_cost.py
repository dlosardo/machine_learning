"""
Perceptron cost function class
HAS A Hypothesis

error matrix:
J(weights) = targets - hypothesis(weights)
number_errors = sum(abs(J(weights)))

"""
from machine_learning.cost_function.cost_function import CostFunction


class PerceptronCost(CostFunction):
    def __init__(self, hypothesis, targets):
        """
        Perceptron cost function
        :param: hypothesis A hypothesis object, e.g., perceptron
        :param: targets A nobs x 1 np array of targets (must be 1s or 0s)
        """
        super(PerceptronCost, self).__init__(hypothesis, targets)

    def get_error_matrix(self):
        """
        Calculates the error matrix as:
            targets - hypothesis(weights)
        :returns: A nobs x 1 np array of error values
        """
        y = self.hypothesis.hypothesis_function(range(0, self.nobs))
        error_matrix = (self.targets - y)
        return error_matrix

    def functional_margin(self, index):
        """
        Confidence in the predicted value
        fm = y_i%*%(weights%*%x_i)
        """
        y = self.targets[index, :].copy()
        y[y == 0] = -1
        fm = y.T.dot(self.hypothesis.features[index, :].dot(self.get_parameters()))
        return fm

    def decision_boundary(self):
        pass
