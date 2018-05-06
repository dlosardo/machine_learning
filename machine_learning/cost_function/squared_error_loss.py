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

Regularization can also be applied. It is an additive function depending
on the type of regularization and the weight. Thus, the final cost
function is:
    cost_function = 1/2*nobs( (Hypothesis - Y)'%*%(Hypothesis - Y) + r(THETA))
where r(THETA) is the regularization function being applied to the parameter
vector. For square error loss, this does not include the intercept parameter.
If no regularizer is requested, then r(THETA) reduces to 0.
"""
from machine_learning.cost_function.cost_function import CostFunction
from numpy import ones, zeros, fill_diagonal, array, hstack


class SquaredErrorLoss(CostFunction):
    def __init__(self, hypothesis, targets, regularizer_name=None,
                 regularization_weight=0):
        """
        Squared error loss cost function
        :param: hypothesis A hypothesis object, e.g., SimpleLinearRegression
        :param: targets A nobs x 1 np array of y values
        :param regularizer_name: A string representing the
            name of the regularizer
        :param regularization_weight: A float of the weight for the regularizer
        """
        super(SquaredErrorLoss, self).__init__(hypothesis, targets,
                                               regularizer_name,
                                               regularization_weight)
        self.set_regularization_matrix()

    def set_regularization_matrix(self):
        """
        Sets the regularization matrix.
        The intercept is not affected in the calculation thus
            the matrix 'picks' out only the
        slope parameters.
        """
        self.regularization_matrix = zeros((self.nparams, self.nparams), float)
        fill_diagonal(self.regularization_matrix, hstack((
            array([0]), ones(self.nparams - 1))))
        self.regularization_matrix = (self.regularization_weight *
                                      self.regularization_matrix)

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
        return ((1./(2.*self.nobs))*(hyp_minus_targets).T.dot(
            hyp_minus_targets) +
                  self.regularizer_cost_function())

    def cost_function_derivative(self):
        """
        The derivative of the cost function for all params
        :returns: A nparam x 1 np array of the values of the
            derivatives of the parameters
        """
        return 1./self.nobs*(self.hypothesis.features.T.dot(
            self.hypothesis_targets()) +
            self.regularizer_cost_function_derivative())

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        return self.convergence_value(current_cost, new_cost) < tolerance

    def convergence_value(self, current_cost, new_cost):
        return abs(current_cost[0] - new_cost[0])
