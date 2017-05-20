from enum import Enum
from machine_learning.hypothesis import simple_linear_regression, multiple_linear_regression, perceptron
from machine_learning.cost_function import squared_error_loss, perceptron_batch_cost, perceptron_online_cost
from machine_learning.algorithm import gradient_descent, stochastic_gradient_descent


class HypothesisTypes(Enum):
    SIMPLE_LINEAR_REGRESSION = 1
    MULTIPLE_LINEAR_REGRESSION = 2
    PERCEPTRON = 3


class HypothesisFactory(object):

    @staticmethod
    def get_hypothesis(n, features, **kwargs):
        if n == HypothesisTypes.SIMPLE_LINEAR_REGRESSION:
            return simple_linear_regression.SimpleLinearRegression(features, **kwargs)
        elif n == HypothesisTypes.MULTIPLE_LINEAR_REGRESSION:
            return multiple_linear_regression.MultipleLinearRegression(features, **kwargs)
        elif n == HypothesisTypes.PERCEPTRON:
            return perceptron.Perceptron(features, **kwargs)

    @staticmethod
    def get_hypothesis_by_name(hypothesis_name, features, **kwargs):
        if hypothesis_name == "simple_linear_regression":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.SIMPLE_LINEAR_REGRESSION, features, **kwargs)
        elif hypothesis_name == "multiple_linear_regression":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.MULTIPLE_LINEAR_REGRESSION, features, **kwargs)
        elif hypothesis_name == "perceptron":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.PERCEPTRON, features, **kwargs)
        else:
            raise ValueError("Invalid hypothesis name: {}".format(hypothesis_name))


class CostFunctionTypes(Enum):
    SQUARED_ERROR_LOSS = 1
    PERCEPTRON_BATCH_COST = 2
    PERCEPTRON_ONLINE_COST = 3


class CostFunctionFactory(object):

    @staticmethod
    def get_cost_function_by_name(cost_function_name, hypothesis, targets, **kwargs):
        if cost_function_name == "squared_error_loss":
            return squared_error_loss.SquaredErrorLoss(hypothesis, targets, **kwargs)
        elif cost_function_name == "perceptron_batch_cost":
            return perceptron_batch_cost.PerceptronBatchCost(hypothesis, targets, **kwargs)
        elif cost_function_name == "perceptron_online_cost":
            return perceptron_online_cost.PerceptronOnlineCost(hypothesis, targets, **kwargs)
        else:
            raise ValueError("Invalid cost function name: {}".format(cost_function_name))


class AlgorithmTypes(Enum):
    GRADIENT_DESCENT = 1
    STOCHASTIC_GRADIENT_DESCENT = 2


class AlgorithmFactory(object):

    @staticmethod
    def get_algorithm_by_name(algorithm_name, cost_function, learning_rate, tolerance, starting_parameter_values, **kwargs):
        if algorithm_name == "gradient_descent":
            return gradient_descent.GradientDescent(cost_function, learning_rate, tolerance, starting_parameter_values, **kwargs)
        elif algorithm_name == "stochastic_gradient_descent":
            return stochastic_gradient_descent.StochasticGradientDescent(cost_function, learning_rate, tolerance, starting_parameter_values, **kwarg)
        else:
            raise ValueError("Invalid algorithm name: {}".format(algorithm_name))
