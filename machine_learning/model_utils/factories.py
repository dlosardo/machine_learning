from enum import Enum
from machine_learning.hypothesis import simple_linear_regression, multiple_linear_regression, logistic_regression, perceptron
from machine_learning.cost_function import squared_error_loss, maximum_likelihood_normal_distribution, log_loss, perceptron_batch_cost, perceptron_online_cost
from machine_learning.algorithm import batch_gradient_descent, stochastic_gradient_descent, newton_raphson


class HypothesisTypes(Enum):
    SIMPLE_LINEAR_REGRESSION = 1
    MULTIPLE_LINEAR_REGRESSION = 2
    LOGISTIC_REGRESSION = 3
    PERCEPTRON = 4


class HypothesisFactory(object):

    @staticmethod
    def get_hypothesis(n, features, **kwargs):
        if n == HypothesisTypes.SIMPLE_LINEAR_REGRESSION:
            return simple_linear_regression.SimpleLinearRegression(features=features, **kwargs)
        elif n == HypothesisTypes.MULTIPLE_LINEAR_REGRESSION:
            return multiple_linear_regression.MultipleLinearRegression(features=features, **kwargs)
        elif n == HypothesisTypes.LOGISTIC_REGRESSION:
            return logistic_regression.LogisticRegression(features=features, **kwargs)
        elif n == HypothesisTypes.PERCEPTRON:
            return perceptron.Perceptron(features=features, **kwargs)

    @staticmethod
    def get_hypothesis_by_name(hypothesis_name, features, **kwargs):
        if hypothesis_name == "simple_linear_regression":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.SIMPLE_LINEAR_REGRESSION, features=features, **kwargs)
        elif hypothesis_name == "multiple_linear_regression":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.MULTIPLE_LINEAR_REGRESSION, features=features, **kwargs)
        elif hypothesis_name == "logistic_regression":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.LOGISTIC_REGRESSION, features=features, **kwargs)
        elif hypothesis_name == "perceptron":
            return HypothesisFactory.get_hypothesis(HypothesisTypes.PERCEPTRON, features=features, **kwargs)
        else:
            raise ValueError("Invalid hypothesis name: {}".format(hypothesis_name))


class CostFunctionTypes(Enum):
    SQUARED_ERROR_LOSS = 1
    MAXIMUM_LIKELIHOOD_NORMAL_DISTRIBUTION = 2
    LOG_LOSS = 3
    PERCEPTRON_BATCH_COST = 4
    PERCEPTRON_ONLINE_COST = 5


class CostFunctionFactory(object):

    @staticmethod
    def get_cost_function_by_name(cost_function_name, hypothesis, targets, **kwargs):
        if cost_function_name == "squared_error_loss":
            return squared_error_loss.SquaredErrorLoss(hypothesis=hypothesis, targets=targets, **kwargs)
        if cost_function_name == "maximum_likelihood_normal_distribution":
            return maximum_likelihood_normal_distribution.MaximumLikelihoodNormalDistribution(hypothesis=hypothesis, targets=targets, **kwargs)
        if cost_function_name == "log_loss":
            return log_loss.LogLoss(hypothesis=hypothesis, targets=targets, **kwargs)
        elif cost_function_name == "perceptron_batch_cost":
            return perceptron_batch_cost.PerceptronBatchCost(hypothesis=hypothesis, targets=targets, **kwargs)
        elif cost_function_name == "perceptron_online_cost":
            return perceptron_online_cost.PerceptronOnlineCost(hypothesis=hypothesis, targets=targets, **kwargs)
        else:
            raise ValueError("Invalid cost function name: {}".format(cost_function_name))


class AlgorithmTypes(Enum):
    BATCH_GRADIENT_DESCENT = 1
    STOCHASTIC_GRADIENT_DESCENT = 2
    NEWTON_RAPHSON = 3


class AlgorithmFactory(object):

    @staticmethod
    def get_algorithm_by_name(algorithm_name, cost_function, **kwargs):
        if algorithm_name == "batch_gradient_descent":
            return batch_gradient_descent.BatchGradientDescent(cost_function=cost_function, **kwargs)
        elif algorithm_name == "stochastic_gradient_descent":
            return stochastic_gradient_descent.StochasticGradientDescent(cost_function=cost_function, **kwargs)
        elif algorithm_name == "newton_raphson":
            return newton_raphson.NewtonRaphson(cost_function=cost_function, **kwargs)
        else:
            raise ValueError("Invalid algorithm name: {}".format(algorithm_name))
