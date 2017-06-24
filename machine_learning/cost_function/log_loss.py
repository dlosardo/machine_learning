"""
Log Loss Cost class function
HAS A Hypothesis
-1/nobs sum(from 1 to nobs) (y_i*log(hypothesis(x_i)) + (1 - y_i)*log(1-hypothesis(x_i)))
in matrix form:
    cost_function = 1/nobs*(-Y'%*%log(Hypothesis) - (1 - Y)'%*%log(1 - Hypothesis)
' is the transpose operation
%*% is matrix multiplication
"""
from machine_learning.cost_function.cost_function import CostFunction
from numpy import dot, ones, log, diag


class LogLoss(CostFunction):
    """log loss loss cost function
    :param hypothesis A hypothesis object, e.g., SimpleLinearRegression
    :param targets A nobs x 1 np array of y values
    """
    def __init__(self, hypothesis, targets):
        super(LogLoss, self).__init__(hypothesis, targets)
        self.nobs = self.targets.shape[0]

    def cost_function(self):
        """Computes the log loss loss cost function
        :returns a 1 x 1 np array containing a float value representing the
         value of the cost function
        """
        ones_vector = ones(self.nobs).reshape(self.nobs, 1)
        cost = (1./(self.nobs))*(-1.*self.targets.T.dot(log(
            self.hypothesis.hypothesis_function())) - (
                ones_vector - self.targets).T.dot(
                    log(ones_vector - self.hypothesis.hypothesis_function())))
        return cost

    def cost_function_derivative(self):
        """The derivative of the cost function for all params
        :returns A nparam x 1 np array of the values of the derivatives of the parameters
        """
        return 1./self.nobs*(self.hypothesis.features.T.dot(
            self.hypothesis.hypothesis_function() - self.targets))

    def cost_function_second_derivative(self):
        """The second derivative of the cost function for all params
        :returns A nparam x nparam np array of the values of the 2nd derivatives of the parameters
        """
        return -1.*(-1./self.nobs * self.hypothesis.features.T.dot(diag(diag(self.hypothesis.hypothesis_function() * (
            1. - self.hypothesis.hypothesis_function())[:,0]))).dot(self.hypothesis.features))

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        return self.convergence_value(current_cost, new_cost) < tolerance

    def convergence_value(self, current_cost, new_cost):
        return abs(current_cost[0] - new_cost[0])
