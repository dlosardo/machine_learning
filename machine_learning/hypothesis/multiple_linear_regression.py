"""
Multiple Linear Regression
IS a Hypothesis
Matrix form:
    hypothesis: X%*%THETA
    where %*% is matrix multiplication, X is a matrix of dimension
    nobs x nparams (nx + 1), and THETA is a matrix
        of dimension nparms (nx + 1) x 1
"""
from machine_learning.hypothesis.regression import Regression
from machine_learning.model_utils.parameter import Parameter


class MultipleLinearRegression(Regression):
    """
    Constructor creates an intercept Parameter object and
    the applicable number of slope Parameter objects
    """
    def __init__(self, features):
        """
        :param features: A nobs x nx numpy array of feature values
        """
        super(MultipleLinearRegression, self).__init__(features)
        self.set_parameters()
        self.error_variance = Parameter(
            name="error_variance", value=None, variance=None,
            default_starting_value=1.)
        self.conditional_mean_parameter_names = (
            self.parameter_list.get_parameter_names())

    def hypothesis_function(self):
        """
        Computes the hypothesis function.
        theta*features
        features are a vector of feature values of dimension nobs x 1
        :returns: A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        return self.features.dot(self.get_parameters())

    def conditional_mean(self):
        """
        Computes the conditional mean of targets given inputs
        If we have y = b*x + e and e ~ N(0, sigma^2)
        then E(Y|X) = b*x
        """
        return self.features.dot(
            self.parameter_list.get_parameter_values_by_name(
                self.conditional_mean_parameter_names))
