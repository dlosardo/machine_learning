"""
Simple Linear Regression
Inherits Hypothesis (IS A Hypothesis)
paramters are intercept and slope. They are of the form:
    2 x 1
features are a single vector of x values of the form:
    nobs x 1
    they are converted to the form:
    nobs x 2
    in the function hypothesis_function where the first row is
    nobs 1s and the second row are values of the x variable
Matrix form:
    hypothesis = X%*%THETA
    where %*% is matrix multiplication, X is a matrix of dimension
    nobs x nparams, and THETA is a matrix of dimension nparms x 1
"""
from machine_learning.hypothesis.hypothesis import Hypothesis
from machine_learning.parameter import Parameter, ParameterList
from numpy import ndarray, array, dot, hstack, reshape, append, ones

class SimpleLinearRegression(Hypothesis):
    """
    Constructor creates two Parameter objects: intercept and slope
    nparams is set to the number of features + 1
    :param features A nobs x 1 numpy array of feature values
    """
    def __init__(self, features):
        super(SimpleLinearRegression, self).__init__(features)
        self.nparams = self.features.shape[1] + 1
        self.intercept = Parameter(name="intercept", value=None, default_starting_value=0.)
        self.slope = Parameter(name="slope", value=None, default_starting_value=0.)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope)
        if self.features.shape[1] != 1:
            raise IncorrectMatrixDimensions(
                "Number of columns is equal to %d but should be equal to 1" % self.features.shape[1])
        # Next line adds a vector of 1s indicating the intercept.
        # self.features becomes a matrix of dimension nobs x 2 with the first column
         # consisting of 1s and the second column consisting of x values.
        self.features = append(ones(self.features.shape[0]).reshape(self.features.shape[0], 1), self.features, 1)

    def update_parameters(self, param_array):
        """
        Updates parameter values
        :param_array A list of parameter values including intercept and slope
         of dimension nparam x 1
        """
        if not self.parameters_initialized():
            raise ParameterValuesNotInitialized(
                    "Parameter values have not yet been initialized")
        if param_array.shape[0] != self.nparams:
            raise IncorrectMatrixDimensions(
                    "Parameter array needs to be %d by 1" % self.nparams)
        self.parameter_list.get_parameter_by_name("intercept").value = param_array[0][0]
        self.parameter_list.get_parameter_by_name("slope").value = param_array[1][0]

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for later computation.
        First horizontally stacks (hstack) intercept and slope
        Next reshapes into an array of dimension number of parameters by 1.
        :returns A numpy array of dimension number of nparams by 1
         containing the values of the parameters.
        """
        if not self.parameters_initialized():
            raise ParameterValuesNotInitialized(
                    "Parameter values have not yet been initialized")
        parms_tmp = hstack((self.parameter_list.get_parameter_by_name("intercept").value
            , self.parameter_list.get_parameter_by_name("slope").value))
        parms_tmp = parms_tmp.reshape(self.nparams, 1)
        return parms_tmp

    def hypothesis_function(self):
        """
        Computes the hypothesis function.
        theta*features
        features are a vector of feature values of dimension nobs x 1
        :returns A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        return self.features.dot(self.get_parameters())


class IncorrectMatrixDimensions(Exception):
    """
    Exception when incorrect matrix dimensions are found
    """

class ParameterValuesNotInitialized(Exception):
    """
    Exception when parameter values are not initialized
    """
