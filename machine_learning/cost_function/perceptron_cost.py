"""
Perceptron cost function class
HAS A Hypothesis

error matrix:
J(weights) = targets - hypothesis(weights)
number_errors = sum(abs(J(weights)))

"""
from machine_learning.cost_function.cost_function import CostFunction


class PerceptronCost(CostFunction):
    """Perceptron cost function
    :param hypothesis A hypothesis object, e.g., perceptron
    :param targets A nobs x 1 np array of targets (must be 1s or 0s)
    """
    def __init__(self, hypothesis, targets):
        super(PerceptronCost, self).__init__(hypothesis, targets)
        self.nobs = self.targets.shape[0]

    def get_error_matrix(self):
        y = self.hypothesis.hypothesis_function(range(0, self.nobs))
        error_matrix = (self.targets - y)
        return error_matrix

    def cost_function_derivative(self):
        # The partial derivative of the cost function with respect to the parameters
        raise NotImplementedError

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        raise NotImplementedError

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
