"""
Maximum Likelihood Function for Normal Distribution (with one error)
Y = X%*%THETA + epsilon
where the conditional probability distribution of Y|X is:
    Y|X ~ N(X%*%THETA, sigma^2)
where epilson, or the errors, are independently distributed as:
    epsilon ~ N(0, sigma^2)
Given this setup, the likelihood is:
    L(Y|X ; THETA) = 1/sigma*sqrt(2PI)*exp(-(Y - X%*%THETA)^2 / 2*sigma^2)
The log-likelihood is:
    l_nobs(Y|X ; THETA) = -nobs/2 * ln(sigma^2) - nobs/2 * ln(2PI) - 1/2*sigma^2 * sum(i to nobs)(Y_i - X_i%*%THETA)^2
"""
from machine_learning.cost_function.cost_function import CostFunction
from numpy import log, pi, var, mean, hstack, vstack, array, sqrt, diag
from numpy.linalg import inv


class MaximumLikelihoodNormalDistribution(CostFunction):
    def __init__(self, hypothesis, targets):
        """
        """
        super(MaximumLikelihoodNormalDistribution, self).__init__(hypothesis, targets)
        self.error_variance = self.hypothesis.error_variance
        self.hypothesis.parameter_list.add_parameter(self.error_variance)
        self.hypothesis.nparams = self.hypothesis.nparams + 1

    def reset(self):
        super(MaximumLikelihoodNormalDistribution, self).reset()
        self.hypothesis.parameter_list.add_parameter(self.error_variance)
        self.hypothesis.nparams = self.hypothesis.nparams + 1

    def maximum_likelihood_fit_function(self):
        #TODO: put the error variance in matrix form such that it is a diagonal matrix, in future multiple errors uncorrelated.
        conditional_mean = self.hypothesis.conditional_mean()
        ml_fit_function = log(self.error_variance.value) - log(var(self.targets)) + 1./self.error_variance.value*var(self.targets) - 1. + ((mean(self.targets) - conditional_mean)).T.dot(1./self.error_variance.value).dot((mean(self.targets) - conditional_mean))
        return ml_fit_function

    def cost_function_derivative(self):
        #TODO: put the error variance in matrix form such that it is a diagonal matrix, in future multiple errors uncorrelated.
        conditional_mean = self.hypothesis.conditional_mean()
        conditional_mean_derivative = (self.hypothesis.features.T.dot(self.targets - conditional_mean))/(self.error_variance.value)
        error_variance_derivative = -1.*self.nobs/(2.*self.error_variance.value) + ((self.targets - conditional_mean).T.dot(self.targets - conditional_mean))/(2.*self.error_variance.value**2)
        return vstack((-1.*conditional_mean_derivative, -1.*error_variance_derivative))

    def cost_function_second_derivative(self):
        #TODO: put the error variance in matrix form such that it is a diagonal matrix, in future multiple errors uncorrelated.
        conditional_mean = self.hypothesis.conditional_mean()
        conditional_mean_covariance = (-1.*self.hypothesis.features.T.dot(self.hypothesis.features))/self.error_variance.value
        #conditional_mean_and_error_variance_covariance = (-1.*self.hypothesis.features.T.dot(self.targets - conditional_mean))/self.error_variance.value**2
        # covariance between error variance and conditional mean is assumed to be zero
        conditional_mean_and_error_variance_covariance = array([[0.], [0.], [0.]])
        error_variance_covariance = (self.nobs)/(2.*self.error_variance.value**2) - ((self.targets - conditional_mean).T.dot(self.targets - conditional_mean))/(self.error_variance.value**3)
        tmp_top=hstack((conditional_mean_covariance, conditional_mean_and_error_variance_covariance))
        tmp_bottom=hstack((conditional_mean_and_error_variance_covariance.T, error_variance_covariance))
        full_matrix = vstack((-1.*tmp_top, -1.*tmp_bottom))
        return full_matrix

    def cost_function(self):
        """Computes the log likelihood
        l_nobs(Y|X ; THETA) = -nobs/2 * ln(sigma^2) - nobs/2 * ln(2PI) - 1/2*sigma^2 * sum(i to nobs)(Y_i - X_i%*%THETA)^2
        """
        conditional_mean = self.hypothesis.conditional_mean()
        log_likelihood = (-1.*self.nobs)/2. * log(self.error_variance.value) - self.nobs/2. * log(2.*pi) - (1.)/(2.*self.error_variance.value) * (self.targets - conditional_mean).T.dot(self.targets - conditional_mean)
        return log_likelihood

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        return self.convergence_value(current_cost, new_cost) < tolerance

    def convergence_value(self, current_cost, new_cost):
        return abs(current_cost[0] - new_cost[0])

    def variance_covariance_matrix(self):
        var_cov_matrix = inv(self.cost_function_second_derivative())
        return var_cov_matrix

    def standard_errors(self):
        return sqrt(diag(self.variance_covariance_matrix()))

    def parameter_variances(self):
        return diag(self.variance_covariance_matrix())
