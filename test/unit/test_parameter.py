"""
Unit Tests for Parameter and ParameterList classes
"""
from nose.tools import assert_raises, assert_is_instance, assert_equals
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy import sqrt, array
from machine_learning.model_utils.parameter import Parameter, ParameterList
from machine_learning.utils.exceptions import (
    ParameterValuesNotInitialized, IncorrectMatrixDimensions)


class TestParameter(object):
    def setup(self):
        self.intercept = Parameter(
            name="intercept", value=None,
            variance=None, default_starting_value=0.)
        self.slope_1 = Parameter(
            name="slope_1", value=None,
            variance=None, default_starting_value=1.)
        self.slope_2 = Parameter(
            name="slope_2", value=None,
            variance=None, default_starting_value=2.)
    """
    Tests
    """
    def test_parameter_instance(self):
        assert_is_instance(self.intercept, Parameter)

    def test_parameter_needs_all_arguments(self):
        assert_raises(TypeError, Parameter, {"name": "intercept", "value": 0.,
                                             "variance": None})
        assert_raises(TypeError, Parameter, {"default_starting_value": 0.,
                                             "value": 0., "variance": None})
        assert_raises(TypeError, Parameter, {"name": "intercept",
                                             "default_starting_value": 0.,
                                             "variance": None})
        assert_raises(TypeError, Parameter, {"name": "intercept",
                                             "default_starting_value": 0.})

    def test_parameter_argument_types(self):
        assert_raises(TypeError, Parameter, {"name": "intercept",
                                             "value": "test", "variance": None,
                                             "default_starting_value": 0.})
        assert_raises(TypeError, Parameter, {"name": 0., "value": None,
                                             "variance": None,
                                             "default_starting_value": 0.})
        assert_raises(TypeError, Parameter, {"name": "intercept",
                                             "value": None, "variance": None,
                                             "default_starting_value": "test"})
        assert_raises(TypeError, Parameter, {"name": "intercept",
                                             "value": None, "variance": "test",
                                             "default_starting_value": 0.})

    def test_parameter_is_initialized(self):
        assert_equals(self.intercept.is_initialized(), False)
        self.intercept.value = 1.
        assert_equals(self.intercept.is_initialized(), True)

    def test_parameter_is_equal(self):
        intercept_1 = Parameter(name="intercept", value=1.,
                                variance=None, default_starting_value=0.)
        assert_equals(self.intercept == self.intercept, True)
        assert_equals(self.intercept == intercept_1, True)
        assert_equals(self.intercept == self.slope_1, False)


class TestParameterList(object):
    def setup(self):
        self.parameter_list = ParameterList()
        self.intercept = Parameter(name="intercept", value=None,
                                   variance=None, default_starting_value=0.)
        self.slope_1 = Parameter(name="slope_1", value=None,
                                 variance=None, default_starting_value=1.)
        self.slope_2 = Parameter(name="slope_2", value=None,
                                 variance=None, default_starting_value=2.)
    """
    Tests
    """
    def test_parameter_list_instance(self):
        assert_is_instance(self.parameter_list, ParameterList)
        with assert_raises(AttributeError) as context:
            self.parameter_list.parameter_list = []
        assert_equals(context.expected, AttributeError)

    def test_parameter_list_is_empty(self):
        assert_equals(self.parameter_list.size, 0)

    def test_parameter_list_add_parameter(self):
        self.parameter_list.add_parameter(self.intercept)
        assert_equals(self.parameter_list.size, 1)
        assert_equals(self.parameter_list.contains_parameter(self.intercept),
                      True)
        assert_raises(TypeError, self.parameter_list.add_parameter, "test")
        assert_raises(Exception, self.parameter_list.add_parameter,
                      self.intercept)

    def test_get_parameter_names(self):
        assert_raises(Exception, self.parameter_list.get_parameter_names)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_equals(self.parameter_list.get_parameter_names(),
                      ['intercept', 'slope_1'])

    def test_get_parameter_by_name(self):
        assert_raises(Exception, self.parameter_list.get_parameter_by_name,
                      "slope")
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_equals(self.parameter_list.get_parameter_by_name("slope_1"),
                      self.slope_1)
        assert_equals(self.parameter_list.get_parameter_by_name("intercept"),
                      self.intercept)
        assert_equals(self.parameter_list.get_parameter_by_name("slope_2"),
                      None)

    def test_all_parameters_initialized(self):
        assert_equals(self.parameter_list.all_parameters_initialized(), False)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        self.parameter_list.parameter_list[0].value = 0.
        assert_equals(self.parameter_list.all_parameters_initialized(), False)
        self.parameter_list.parameter_list[1].value = 0.
        assert_equals(self.parameter_list.all_parameters_initialized(), True)

    def test_initialize_parameters(self):
        assert_equals(self.parameter_list.initialize_parameters(), None)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        self.parameter_list.initialize_parameters()
        assert_equals(
            self.parameter_list.get_parameter_by_name("intercept").value, 0.)
        assert_equals(
            self.parameter_list.get_parameter_by_name("slope_1").value, 1.)
        self.parameter_list.initialize_parameters({'intercept': 8.})
        assert_equals(
            self.parameter_list.get_parameter_by_name("intercept").value, 8.)

    def test_set_parameter_variances(self):
        assert_raises(Exception,
                      self.parameter_list.set_parameter_variances, [3.])
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        parameter_variance_list = [3., 4.]
        self.parameter_list.set_parameter_variances(parameter_variance_list)
        assert_equals(
            self.parameter_list.get_parameter_by_name("intercept").variance,
            3.)
        assert_equals(
            self.parameter_list.get_parameter_by_name("slope_1").variance, 4.)
        assert_raises(
            TypeError, self.parameter_list.set_parameter_variances, [0.])
        assert_raises(
            TypeError, self.parameter_list.set_parameter_variances, 0.)

    def test_set_covariance_matrix(self):
        true_cov_array = array([[1.]])
        assert_equals(
            self.parameter_list.set_covariance_matrix(array([1.])), None)
        self.parameter_list.add_parameter(self.intercept)
        assert_raises(TypeError, self.parameter_list.set_covariance_matrix, [])
        assert_raises(IncorrectMatrixDimensions,
                      self.parameter_list.set_covariance_matrix, array([1.]))
        assert_raises(IncorrectMatrixDimensions,
                      self.parameter_list.set_covariance_matrix,
                      array([[1.], [1.]]))
        self.parameter_list.set_covariance_matrix(true_cov_array)
        assert_array_equal(true_cov_array,
                           self.parameter_list.parameter_covariance_matrix)

    def test_update_parameters(self):
        assert_raises(Exception,
                      self.parameter_list.update_parameters, array([[1.]]))
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(ParameterValuesNotInitialized,
                      self.parameter_list.update_parameters,
                      array([[1., 1.], [1., 1.]]))
        self.parameter_list.initialize_parameters()
        assert_raises(TypeError, self.parameter_list.update_parameters, "")
        assert_raises(
            TypeError, self.parameter_list.update_parameters, array([1.]))
        assert_raises(
            TypeError, self.parameter_list.update_parameters,
            array([[1., 1.], [1., 1.]]))
        accepted_array = array([[1.], [1.]])
        self.parameter_list.update_parameters(accepted_array)
        assert_array_equal(
            self.parameter_list.get_parameters(), accepted_array)

    def test_get_parameter_at_index(self):
        assert_raises(Exception, self.parameter_list.get_parameter_at_index, 0)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            IndexError, self.parameter_list.get_parameter_at_index, 10)
        assert_raises(
            IndexError, self.parameter_list.get_parameter_at_index, -1)
        assert_equals(
            self.intercept, self.parameter_list.get_parameter_at_index(0))
        assert_equals(
            self.slope_1, self.parameter_list.get_parameter_at_index(1))

    def test_get_parameter_values_by_name(self):
        assert_raises(Exception,
                      self.parameter_list.get_parameter_values_by_name,
                      ["intercept"])
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            Exception, self.parameter_list.get_parameter_values_by_name,
            ["slope_2"])
        assert_raises(
            Exception, self.parameter_list.get_parameter_values_by_name,
            ["slope_1"])
        self.parameter_list.initialize_parameters()
        assert_array_equal(
            array([[0.]]),
            self.parameter_list.get_parameter_values_by_name(["intercept"]))
        assert_array_equal(
            array([[0.], [1.]]),
            self.parameter_list.get_parameter_values_by_name(
                ["intercept", "slope_1"]))
        assert_array_equal(
            array([[1.]]),
            self.parameter_list.get_parameter_values_by_name(
                ["slope_1", "slope_2"]))

    def test_get_parameter_values_not_in_list(self):
        assert_raises(
            Exception, self.parameter_list.get_parameter_values_not_in_list,
            ["intercept"])
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        self.parameter_list.add_parameter(self.slope_2)
        assert_raises(
            Exception, self.parameter_list.get_parameter_values_not_in_list,
            [self.intercept, self.slope_1])
        self.parameter_list.initialize_parameters()
        assert_raises(
            Exception, self.parameter_list.get_parameter_values_not_in_list,
            [self.intercept, self.slope_1, self.slope_2])
        assert_array_equal(
            array([[0.]]),
            self.parameter_list.get_parameter_values_not_in_list(
                [self.slope_1, self.slope_2]))

    def test_get_parameter_variances(self):
        assert_raises(Exception, self.parameter_list.get_parameter_variances)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_array_equal(
            array([[None], [None]]),
            self.parameter_list.get_parameter_variances())
        self.parameter_list.set_parameter_variances([2., 3.])
        assert_array_equal(
            array([[2.], [3.]]),
            self.parameter_list.get_parameter_variances())

    def test_get_parameter_standard_errors(self):
        assert_raises(
            Exception, self.parameter_list.get_parameter_standard_errors)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_array_equal(
            array([[None], [None]]),
            self.parameter_list.get_parameter_standard_errors())
        self.parameter_list.set_parameter_variances([2., 3.])
        assert_array_almost_equal(
            array([[sqrt(2.)], [sqrt(3.)]]),
            self.parameter_list.get_parameter_standard_errors())

    def test_contains_parameter(self):
        assert_equals(
            False, self.parameter_list.contains_parameter(self.intercept))
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            TypeError, self.parameter_list.contains_parameter, "slope_1")
        assert_equals(
            True, self.parameter_list.contains_parameter(self.intercept))
        assert_equals(
            False, self.parameter_list.contains_parameter(self.slope_2))

    def test_contains_parameter_by_name(self):
        assert_equals(
            False, self.parameter_list.contains_parameter_by_name("intercept"))
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            TypeError, self.parameter_list.contains_parameter_by_name,
            self.intercept)
        assert_equals(
            True, self.parameter_list.contains_parameter_by_name("intercept"))
        assert_equals(
            False, self.parameter_list.contains_parameter_by_name("slope_2"))

    def test_parameter_index(self):
        assert_raises(
            Exception, self.parameter_list.parameter_index, self.intercept)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            TypeError, self.parameter_list.parameter_index, "intercept")
        assert_equals(0, self.parameter_list.parameter_index(self.intercept))
        assert_equals(-1, self.parameter_list.parameter_index(self.slope_2))

    def test_remove_parameter_from_parameter_object(self):
        assert_raises(
            Exception,
            self.parameter_list.remove_parameter_from_parameter_object,
            self.intercept)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            TypeError,
            self.parameter_list.remove_parameter_from_parameter_object,
            "intercept")
        assert_raises(
            Exception,
            self.parameter_list.remove_parameter_from_parameter_object,
            self.slope_2)
        self.parameter_list.remove_parameter_from_parameter_object(
            self.intercept)
        assert_equals(1, self.parameter_list.size)
        assert_equals(
            False, self.parameter_list.contains_parameter(self.intercept))
        assert_equals(
            True, self.parameter_list.contains_parameter(self.slope_1))

    def test_clear_parameter_list(self):
        assert_raises(Exception, self.parameter_list.clear_parameter_list)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        self.parameter_list.clear_parameter_list()
        assert_equals(0, self.parameter_list.size)
        assert_equals(
            False, self.parameter_list.contains_parameter(self.intercept))
        assert_equals(
            False, self.parameter_list.contains_parameter(self.slope_1))

    def test_get_unique_elements_in_covariance_matrix(self):
        assert_equals(
            0, self.parameter_list.get_unique_elements_in_covariance_matrix())
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_equals(
            3, self.parameter_list.get_unique_elements_in_covariance_matrix())

    def test_get_parameters(self):
        assert_raises(Exception, self.parameter_list.get_parameters)
        self.parameter_list.add_parameter(self.intercept)
        self.parameter_list.add_parameter(self.slope_1)
        assert_raises(
            ParameterValuesNotInitialized, self.parameter_list.get_parameters)
        self.parameter_list.initialize_parameters()
        assert_array_equal(
            array([[0.], [1.]]), self.parameter_list.get_parameters())
