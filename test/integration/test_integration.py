"""
Integration Tests
"""

from nose.tools import assert_raises, assert_is_instance, assert_equals, assert_almost_equals
from machine_learning import runmain


class TestSimpleLinearRegression(object):

    def setup(self):
        print("setup() before any methods in this class")
        pass

    """
    Tests
    """
    def test_simple_linear_regression_run(self):
        pass

    def test_multiple_linear_regression_run(self):
        pass

    def test_percetron_batch_run(self):
        pass

    def test_perceptron_online_run(self):
        pass

    def test_logistic_regression_run(self):
        inputs = ["--input-data-file", "data/input/sample_logistic.csv", "--number-features", '1', "--number-targets", '1', "--hypothesis-name", 'logistic_regression', "--cost-function-name", 'logistic_regression_cost', "--algorithm-name", 'newton_raphson', "--tolerance", '.0001']
        parameters = runmain.main(inputs)
        assert_almost_equals(parameters[0,0], -0.62494949, places=8)
        assert_almost_equals(parameters[1,0], 0.0128923, places=7)
