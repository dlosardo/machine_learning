"""
A regression hypothesis
parameters are intercept and nx slopes, where nx is the number of features (independent variables) of the form:
    (nx + 1) x 1
features are a matrix of x values of the form:
    nobs x nx
    converted to the form:
    nobs x (nx + 1)
    to include the intercept.
"""
from machine_learning.hypothesis.hypothesis import Hypothesis
from machine_learning.model_utils.parameter import Parameter
from machine_learning.utils.math_utils import add_constant


class Regression(Hypothesis):
    """
    """
    def __init__(self, features):
        super(Regression, self).__init__(features)
        # Next line adds a vector of 1s indicating the intercept.
        # self.features becomes a matrix of dimension nobs x nx with the first column
         # consisting of 1s and the next columns consisting of x values.
        self.features = add_constant(self.features)

    def set_parameters(self):
        self.nparams = self.nfeatures + 1
        intercept = Parameter(name="intercept", value=None, variance=None, default_starting_value=0.)
        self.parameter_list.add_parameter(intercept)
        slope_names = []
        for i in range(0, self.nparams - 1):
            slope_name = "slope_{}".format(i)
            tmp_slope = Parameter(name=slope_name, value=None, variance=None, default_starting_value=0.)
            slope_names.append(slope_name)
            self.parameter_list.add_parameter(tmp_slope)
