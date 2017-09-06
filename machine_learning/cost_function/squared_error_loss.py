"""
Squared error loss class function
HAS A Hypothesis
1/n*nobs sum(from 1 to nobs) (hypothesis_i - y_i)^2
in matrix form, for linear regression:
    cost_function = 1/2*nobs(X%*%THETA - Y)'%*%(X%*%THETA - Y)
in matrix form in general:
    cost_function = 1/2*nobs(Hypothesis - Y)'%*%(Hypothesis - Y)
' is the transpose operation
%*% is matrix multiplication
"""
from machine_learning.cost_function.cost_function import CostFunction
from numpy import ones


class SquaredErrorLoss(CostFunction):
    def __init__(self, hypothesis, targets):
        """
        Squared error loss cost function
        :param: hypothesis A hypothesis object, e.g., SimpleLinearRegression
        :param: targets A nobs x 1 np array of y values
        """
        super(SquaredErrorLoss, self).__init__(hypothesis, targets)

    def hypothesis_targets(self):
        """
        Computes the hypothesis function - targets
        :returns: an np array of dimension nobs x 1
        """
        return self.hypothesis.hypothesis_function() - self.targets

    def cost_function(self):
        """
        Computes the squared error loss cost function
        :returns: a 1 x 1 np array containing a float value representing the
         value of the cost function
        """
        hyp_minus_targets = self.hypothesis_targets()
        return (1./(2.*self.nobs))*(hyp_minus_targets).T.dot(hyp_minus_targets)

    def cost_function_derivative(self):
        """
        The derivative of the cost function for all params
        :returns: A nparam x 1 np array of the values of the derivatives of the parameters
        """
        return 1./self.nobs*(self.hypothesis.features.T.dot(self.hypothesis_targets()))

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        return self.convergence_value(current_cost, new_cost) < tolerance

    def convergence_value(self, current_cost, new_cost):
        return abs(current_cost[0] - new_cost[0])

    def cost_function_tmp(self):
        """
        Computes the cost function the long way - not using hypothesis
        """
        return self.targets.T.dot(self.targets) - self.targets.T.dot(self.hypothesis.features
                ).dot(self.get_parameters()) - self.get_parameters().T.dot(self.hypothesis.features.T
                        ).dot(self.targets) + self.get_parameters().T.dot(self.hypothesis.features.T
                                ).dot(self.hypothesis.features).dot(self.get_parameters())

    def cost_function_derivative_int(self):
        """
        The derivative of the cost function wrt intercept param
        :returns: A 1 x 1 np array of the value of the derivative of the cost function
         wrt the intercept param
        """
        return 1./self.nobs*ones(self.nobs).reshape(1, self.nobs).dot(self.hypothesis_targets())

    def cost_function_derivative_slope(self):
        """
        The derivative of the cost function wrt slope param
        :returns: A 1 x 1 np array of the value of the derivative of the cost function
         wrt the slope param
        """
        return 1./self.nobs*(self.hypothesis_targets()).T.dot(self.hypothesis.features)
