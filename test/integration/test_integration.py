"""
Integration Tests
"""

from nose.tools import assert_almost_equals
from machine_learning import runmain

SIMPLE_LINEAR_REGRESSION_INTERCEPT = -3.89026285
SIMPLE_LINEAR_REGRESSION_SLOPE = 1.1924793

MULTIPLE_LINEAR_REGRESSION_INTERCEPT = 101.21205399
MULTIPLE_LINEAR_REGRESSION_SLOPE_1 = 1.00059136
MULTIPLE_LINEAR_REGRESSION_SLOPE_2 = 1.07096345
MULTIPLE_LINEAR_REGRESSION_RESIDUAL_VARIANCE = 36.49787789

MULTIPLE_LINEAR_RIDGE_REGRESSION_INTERCEPT = 101.28964862
MULTIPLE_LINEAR_RIDGE_REGRESSION_SLOPE_1 = 0.99954165
MULTIPLE_LINEAR_RIDGE_REGRESSION_SLOPE_2 = 1.06966768

LOGISTIC_REGRESSION_INTERCEPT = -0.62494949
LOGISTIC_REGRESSION_SLOPE = 0.0128923

PERCEPTRON_ONLINE_BIAS = -1.0000
PERCEPTRON_ONLINE_FEATURE_1 = 2.351437
PERCEPTRON_ONLINE_FEATURE_2 = -1.514955

PERCEPTRON_BATCH_BIAS = 0.0000
PERCEPTRON_BATCH_FEATURE_1 = 0.22535731
PERCEPTRON_BATCH_FEATURE_2 = -0.1607145


class TestIntegration(object):

    def setup(self):
        print("setup() before any methods in this class")
        pass

    """
    Integration Tests
    """
    def test_simple_linear_regression_run(self):
        inputs = ["--input-data-file", "data/input/ex1data1.txt",
                  "--number-features", '1', "--number-targets", '1',
                  "--hypothesis-name", 'simple_linear_regression',
                  "--cost-function-name", 'squared_error_loss',
                  "--algorithm-name", "batch_gradient_descent",
                  "--learning-rate", "0.001", "--tolerance", '.000000001']
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0],
                             SIMPLE_LINEAR_REGRESSION_INTERCEPT, places=4)
        assert_almost_equals(parameters[1, 0],
                             SIMPLE_LINEAR_REGRESSION_SLOPE, places=4)

    def test_multiple_linear_regression_squared_error_loss_batch_gradient_descent_run(self):
        inputs = ["--input-data-file", "data/input/sample_data.csv",
                  "--number-features", '2', "--number-targets", '1',
                  "--hypothesis-name", 'multiple_linear_regression',
                  "--cost-function-name", 'squared_error_loss',
                  "--algorithm-name", "batch_gradient_descent",
                  "--learning-rate", "0.0001", "--tolerance", '.0000000001']
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0],
                             MULTIPLE_LINEAR_REGRESSION_INTERCEPT, places=4)
        assert_almost_equals(parameters[1, 0],
                             MULTIPLE_LINEAR_REGRESSION_SLOPE_1, places=4)
        assert_almost_equals(parameters[2, 0],
                             MULTIPLE_LINEAR_REGRESSION_SLOPE_2, places=4)

    def test_multiple_linear_ridge_regression_squared_error_loss_batch_gradient_descent_run(self):
        inputs = ["--input-data-file", "data/input/sample_data.csv",
                  "--number-features", '2', "--number-targets", '1',
                  "--hypothesis-name", 'multiple_linear_regression',
                  "--cost-function-name", 'squared_error_loss',
                  "--algorithm-name", "batch_gradient_descent",
                  "--learning-rate", "0.0001", "--tolerance", '.0000000001',
                  "--regularizer-name", "ridge",
                  "--regularization-weight", "5"]
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0],
                             MULTIPLE_LINEAR_RIDGE_REGRESSION_INTERCEPT,
                             places=4)
        assert_almost_equals(parameters[1, 0],
                             MULTIPLE_LINEAR_RIDGE_REGRESSION_SLOPE_1,
                             places=4)
        assert_almost_equals(parameters[2, 0],
                             MULTIPLE_LINEAR_RIDGE_REGRESSION_SLOPE_2,
                             places=4)

    """
    def test_multiple_linear_regression_maximum_likelihood_batch_gradient_descent_run(self):
        inputs = ["--input-data-file", "data/input/sample_data.csv",
                  "--number-features", '2', "--number-targets", '1',
                  "--hypothesis-name", 'multiple_linear_regression',
                  "--cost-function-name",
                  'maximum_likelihood_normal_distribution',
                  "--algorithm-name", "batch_gradient_descent",
                  "--learning-rate", "0.0001", "--tolerance", '.0000000001']
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0],
                             MULTIPLE_LINEAR_REGRESSION_INTERCEPT, places=1)
        assert_almost_equals(parameters[1, 0],
                             MULTIPLE_LINEAR_REGRESSION_SLOPE_1, places=1)
        assert_almost_equals(parameters[2, 0],
                             MULTIPLE_LINEAR_REGRESSION_SLOPE_2, places=1)
        assert_almost_equals(parameters[3, 0],
                             MULTIPLE_LINEAR_REGRESSION_RESIDUAL_VARIANCE,
                             places=1)
    """

    def test_percetron_batch_run(self):
        inputs = ["--input-data-file", "data/input/sample_perceptron.csv",
                  "--number-features", '2', "--number-targets", '1',
                  "--hypothesis-name", "perceptron",
                  "--cost-function-name", "perceptron_batch_cost",
                  "--algorithm-name", "batch_gradient_descent",
                  "--learning-rate", '1.']
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0], PERCEPTRON_BATCH_BIAS, places=4)
        assert_almost_equals(parameters[1, 0],
                             PERCEPTRON_BATCH_FEATURE_1, places=4)
        assert_almost_equals(parameters[2, 0],
                             PERCEPTRON_BATCH_FEATURE_2, places=4)

    def test_perceptron_online_run(self):
        inputs = ["--input-data-file", "data/input/sample_perceptron.csv",
                  "--number-features", '2', "--number-targets", '1',
                  "--hypothesis-name", "perceptron",
                  "--cost-function-name", "perceptron_online_cost",
                  "--algorithm-name", "stochastic_gradient_descent",
                  "--learning-rate", '1.']
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0],
                             PERCEPTRON_ONLINE_BIAS, places=4)
        assert_almost_equals(parameters[1, 0],
                             PERCEPTRON_ONLINE_FEATURE_1, places=4)
        assert_almost_equals(parameters[2, 0],
                             PERCEPTRON_ONLINE_FEATURE_2, places=4)

    def test_logistic_regression_newton_raphson_run(self):
        inputs = ["--input-data-file", "data/input/sample_logistic.csv",
                  "--number-features", '1', "--number-targets", '1',
                  "--hypothesis-name", 'logistic_regression',
                  "--cost-function-name", 'log_loss',
                  "--algorithm-name", 'newton_raphson',
                  "--tolerance", '.0001']
        learning_model_obj = runmain.main(inputs)
        parameters = learning_model_obj.get_parameter_point_estimates()
        assert_almost_equals(parameters[0, 0],
                             LOGISTIC_REGRESSION_INTERCEPT, places=4)
        assert_almost_equals(parameters[1, 0],
                             LOGISTIC_REGRESSION_SLOPE, places=4)
