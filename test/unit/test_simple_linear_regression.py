"""
Unit Tests for simple linear regression class
"""

from nose.tools import assert_raises, assert_is_instance, assert_equals
from machine_learning.hypothesis.simple_linear_regression import (
    SimpleLinearRegression)
from machine_learning.utils.exceptions import (
    IncorrectMatrixDimensions, ParameterValuesNotInitialized)
from numpy import array

INTERCEPT_VALUE = 0.
SLOPE_VALUE = 0.
PARAM_DICT = {"intercept": INTERCEPT_VALUE,
              "slope": SLOPE_VALUE}
PARAM_ARRAY = array([[INTERCEPT_VALUE], [SLOPE_VALUE]])  # 2 x 1
PARAM_ARRAY_INCORRECT = array([INTERCEPT_VALUE, SLOPE_VALUE])
FEATURES_INCORRECT = array([[1], [1]])
FEATURES = array([1, 2, 3]).reshape(3, 1)


class TestSimpleLinearRegression(object):

    def setup(self):
        print("setup() before any methods in this class")
        self.hypo = SimpleLinearRegression(FEATURES)

    """
    Tests
    """
    def test_simple_linear_regression_instance(self):
        assert_is_instance(self.hypo, SimpleLinearRegression)

    def test_initialize_parameters_correctly_sets_parameter_estimates(self):
        self.hypo.initialize_parameters(PARAM_DICT)
        assert_equals(self.hypo.intercept.value, INTERCEPT_VALUE)
        assert_equals(self.hypo.slope.value, SLOPE_VALUE)

    def test_update_parameters_raises_error_without_initialization(self):
        assert_raises(ParameterValuesNotInitialized,
                      self.hypo.update_parameters, PARAM_ARRAY)

    def test_update_parameters_raises_incorrect_matrix_dimensions(self):
        self.hypo.initialize_parameters()
        assert_raises(IncorrectMatrixDimensions,
                      self.hypo.update_parameters, PARAM_ARRAY_INCORRECT)
