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
from hypothesis import Hypothesis
from parameter import Parameter
from numpy import ndarray, array, dot, hstack, reshape, append, ones

class SimpleLinearRegression(Hypothesis):
    """
    Constructor creates two Parameter objects: intercept and slope
    nparams is set to 2
    """
    def __init__(self):
        super(SimpleLinearRegression, self).__init__()
        self.intercept = Parameter(name="intercept", value=None)
        self.slope = Parameter(name="slope", value=None)
        self.nparams = 2

    def parameters_initialized(self):
        """Checks to see whether parameters have been initialized.
        :return True if parameters have been initialized, false otherwise
        """
        return self.intercept.is_initialized() and self.slope.is_initialized()

    def initialize_parameters(self, param_dict):
        """
        initializes parameter values
        :param param_values A dict of parameter_name: parameter_value
        """
        self.intercept.value = param_dict["intercept"]
        self.slope.value = param_dict["slope"]

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
        self.intercept.value = param_array[0][0]
        self.slope.value = param_array[1][0]

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for later computation.
        First horizontally stacks (hstack) intercept and slope
        Next reshapes into an array of dimension number of parameters
        by 1.
        :returns A numpy array of dimension number of nparams by 1
         containing the values of the parameters.
        """
        if not self.parameters_initialized():
            raise ParameterValuesNotInitialized(
                    "Parameter values have not yet been initialized")
        parms_tmp = hstack((self.intercept.value, self.slope.value))
        parms_tmp = parms_tmp.reshape(self.nparams, 1)
        return parms_tmp

    def feature_setup(self, features):
        """
        Reshapes features into suitable form for later computation.
        Adds a vector of 1s indicating the intercept.
        :param features A vector of x values of dimension 1 x nobs
        :returns A matrix of dimension nobs x 2 with the first column
         consisting of 1s and the second column consisting of x values.
        """
        if features.shape[1] != 1:
            raise IncorrectMatrixDimensions(
                "Number of rows is equal to %d but should be equal to 1" % features.shape[0])
        return append(ones(features.shape[0]).reshape(features.shape[0], 1), features, 1)

    def hypothesis_function(self, features):
        """
        Computes the hypothesis function.
        theta*features
        :param features A vector of feature values of dimension 1 x nobs
        :returns A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        features_ = self.feature_setup(features)
        return features_.dot(self.get_parameters())


class IncorrectMatrixDimensions(Exception):
    """
    Exception when incorrect matrix dimensions are found
    """

class ParameterValuesNotInitialized(Exception):
    """
    Exception when parameter values are not initialized
    """
