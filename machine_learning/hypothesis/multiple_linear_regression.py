"""
Multiple Linear Regression
IS a Hypothesis
parameters are intercept and nx slopes, where nx is the number of features (independent variables) of the form:
    (nx + 1) x 1
features are a matrix of x values of the form:
    nobs x nx
    converted to the form:
    nobs x (nx + 1)
    to include the intercept.
Matrix form:
    hypothesis: X%*%THETA
    where %*% is matrix multiplication, X is a matrix of dimension
    nobs x nparams (nx + 1), and THETA is a matrix of dimension nparms (nx + 1) x 1
"""

from machine_learning.hypothesis.hypothesis import Hypothesis
from machine_learning.model_utils.parameter import Parameter, ParameterList
from numpy import ndarray, array, dot, hstack, reshape, append, ones


class MultipleLinearRegression(Hypothesis):
    """
    Constructor creates an intercept Parameter object and
    the applicable number of slope Parameter objects
    """
    def __init__(self, features):
        super(MultipleLinearRegression, self).__init__(features)
        """
        We know we have an intercept but not sure how many slope parameters need to be created yet
        :param features A nobs x nx numpy array of feature values
        """
        self.nparams = features.shape[1] + 1
        self.intercept = Parameter(name="intercept", value=None, default_starting_value=0.)
        self.parameter_list.add_parameter(self.intercept)
        for i in range(0, features.shape[1]):
            tmp_slope = Parameter(name="slope_{}".format(i), value=None, default_starting_value=0.)
            self.parameter_list.add_parameter(tmp_slope)
        if self.features.shape[1] != self.nparams - 1:
            raise IncorrectMatrixDimensions(
                "Number of columns is equal to %d but should be equal to %d" % self.features.shape[1], self.nparams - 1)
        # Next line adds a vector of 1s indicating the intercept.
        # self.features becomes a matrix of dimension nobs x nx with the first column
         # consisting of 1s and the next columns consisting of x values.
        self.features = append(ones(self.features.shape[0]).reshape(self.features.shape[0], 1), self.features, 1)

    def hypothesis_function(self):
        """
        Computes the hypothesis function.
        theta*features
        features are a vector of feature values of dimension nobs x 1
        :returns A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        return self.features.dot(self.get_parameters())
