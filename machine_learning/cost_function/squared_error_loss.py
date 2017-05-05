"""
Squared error loss class function
HAS A Hypothesis
1/n*nobs sum(from 1 to nobs) (hypothesis_i - y_i)^2
in matrix form:
    cost_function = 1/2*nobs(X%*%THETA - Y)'%*%(X%*%THETA - Y)
' is the transpose operation
%*% is matrix multiplication
"""
from machine_learning.cost_function.cost_function import CostFunction
from numpy import dot, ones

class SquaredErrorLoss(CostFunction):
    """Squared error loss cost function
    :param hypothesis A hypothesis object, e.g., SimpleLinearRegression
    :param targets A nobs x 1 np array of y values
    """
    def __init__(self, hypothesis, targets):
        super(SquaredErrorLoss, self).__init__(hypothesis, targets)
        self.nobs = self.targets.shape[0]

    def initialize_parameters(self, param_dict=None):
        """Initializes the parameter values
        :param param_dict A dictionary with the form parameter_name: parameter_value
        """
        self.hypothesis.initialize_parameters(param_dict)

    def update_parameters(self, param_array):
        """Updates parameter values
        :param param_array A numpy array of dimension nparams x 1 consisting of
         parameter values.
        """
        self.hypothesis.update_parameters(param_array)

    def get_parameters(self):
        """Gets parameter values
        :return A nparam x 1 numpy array of parameter values
        """
        return self.hypothesis.get_parameters()

    def hypothesis_targets(self):
        """Computes the hypothesis function - targets
        :returns an np array of dimension nobs x 1
        """
        return self.hypothesis.hypothesis_function() - self.targets

    def cost_function(self):
        """Computes the squared error loss cost function
        :returns a 1 x 1 np array containing a float value representing the
         value of the cost function
        """
        hyp_minus_targets = self.hypothesis_targets()
        return (1./(2.*self.nobs))*(hyp_minus_targets).T.dot(hyp_minus_targets)

    def cost_function_derivative(self):
        """The derivative of the cost function for all params
        :returns A nparam x 1 np array of the values of the derivatives of the parameters
        """
        return 1./self.nobs*(self.hypothesis.features.T.dot(self.hypothesis_targets()))

    def cost_function_tmp(self):
        """Computes the cost function the long way - not using hypothesis
        """
        return self.targets.T.dot(self.targets) - self.targets.T.dot(self.hypothesis.features
                ).dot(self.get_parameters()) - self.get_parameters().T.dot(self.hypothesis.features.T
                        ).dot(self.targets) + self.get_parameters().T.dot(self.hypothesis.features.T
                                ).dot(self.hypothesis.features).dot(self.get_parameters())

    def cost_function_derivative_int(self):
        """The derivative of the cost function wrt intercept param
        :returns A 1 x 1 np array of the value of the derivative of the cost function
         wrt the intercept param
        """
        return 1./self.nobs*ones(self.nobs).reshape(1, self.nobs).dot(self.hypothesis_targets())

    def cost_function_derivative_slope(self):
        """The derivative of the cost function wrt slope param
        :returns A 1 x 1 np array of the value of the derivative of the cost function
         wrt the slope param
        """
        return 1./self.nobs*(self.hypothesis_targets()).T.dot(self.hypothesis.features)
