from enum import Enum
from machine_learning.hypothesis import (
    simple_linear_regression, multiple_linear_regression,
    logistic_regression, perceptron)
from machine_learning.cost_function import (
    squared_error_loss, maximum_likelihood_normal_distribution,
    log_loss, perceptron_batch_cost, perceptron_online_cost)
from machine_learning.algorithm import (
    batch_gradient_descent, stochastic_gradient_descent,
    newton_raphson)


class Types(Enum):

    @classmethod
    def values_list(cls):
        return [type_.value for type_ in list(cls)]

    @classmethod
    def names_list(cls):
        return [type_.name.lower() for type_ in list(cls)]

    @classmethod
    def tuple_pair(cls):
        return list(zip(map(lambda x: str(x),
                            cls.values_list()),
                        map(lambda x: x.replace("_", " "),
                            cls.names_list())))

    @classmethod
    def get_type_from_number(cls, number):
        return [i for i in cls
                if i.value == number][0]


class HypothesisTypes(Types):
    SIMPLE_LINEAR_REGRESSION = 1
    MULTIPLE_LINEAR_REGRESSION = 2
    LOGISTIC_REGRESSION = 3
    PERCEPTRON = 4


class HypothesisFactory(object):

    @staticmethod
    def get_hypothesis(n, features, **kwargs):
        if n == HypothesisTypes.SIMPLE_LINEAR_REGRESSION.value:
            return simple_linear_regression.SimpleLinearRegression(
                features=features, **kwargs)
        elif n == HypothesisTypes.MULTIPLE_LINEAR_REGRESSION.value:
            return multiple_linear_regression.MultipleLinearRegression(
                features=features, **kwargs)
        elif n == HypothesisTypes.LOGISTIC_REGRESSION.value:
            return logistic_regression.LogisticRegression(
                features=features, **kwargs)
        elif n == HypothesisTypes.PERCEPTRON.value:
            return perceptron.Perceptron(features=features, **kwargs)
        else:
            raise ValueError("Invalid hypothesis id: {}".format(n))

    @staticmethod
    def get_hypothesis_by_name(hypothesis_name, features, **kwargs):
        if hypothesis_name == "simple_linear_regression":
            return HypothesisFactory.get_hypothesis(
                HypothesisTypes.SIMPLE_LINEAR_REGRESSION,
                features=features, **kwargs)
        elif hypothesis_name == "multiple_linear_regression":
            return HypothesisFactory.get_hypothesis(
                HypothesisTypes.MULTIPLE_LINEAR_REGRESSION,
                features=features, **kwargs)
        elif hypothesis_name == "logistic_regression":
            return HypothesisFactory.get_hypothesis(
                HypothesisTypes.LOGISTIC_REGRESSION,
                features=features, **kwargs)
        elif hypothesis_name == "perceptron":
            return HypothesisFactory.get_hypothesis(
                HypothesisTypes.PERCEPTRON, features=features, **kwargs)
        else:
            raise ValueError("Invalid hypothesis name: {}".format(
                hypothesis_name))


class RegularizerTypes(Types):
    NONE = 1
    RIDGE = 2
    LASSO = 3
    ELASTIC_NET = 4


class CostFunctionTypes(Types):
    SQUARED_ERROR_LOSS = 1
    MAXIMUM_LIKELIHOOD_NORMAL_DISTRIBUTION = 2
    LOG_LOSS = 3
    PERCEPTRON_BATCH_COST = 4
    PERCEPTRON_ONLINE_COST = 5


class HypothesisCostFunctions(object):
    @staticmethod
    def cost_function_hypothesis_dict():
        return {HypothesisTypes.SIMPLE_LINEAR_REGRESSION: set(
            [CostFunctionTypes.SQUARED_ERROR_LOSS]),
                HypothesisTypes.MULTIPLE_LINEAR_REGRESSION: set(
                    [CostFunctionTypes.SQUARED_ERROR_LOSS,
                     CostFunctionTypes.MAXIMUM_LIKELIHOOD_NORMAL_DISTRIBUTION]
                ),
                HypothesisTypes.LOGISTIC_REGRESSION: set(
                    [CostFunctionTypes.LOG_LOSS]),
                HypothesisTypes.PERCEPTRON: set(
                    [CostFunctionTypes.PERCEPTRON_BATCH_COST,
                     CostFunctionTypes.PERCEPTRON_ONLINE_COST])
               }


class CostFunctionFactory(object):

    @staticmethod
    def get_cost_function(n, hypothesis, targets, **kwargs):
        if n == CostFunctionTypes.SQUARED_ERROR_LOSS.value:
            return squared_error_loss.SquaredErrorLoss(
                hypothesis=hypothesis, targets=targets, **kwargs)
        if n == CostFunctionTypes.MAXIMUM_LIKELIHOOD_NORMAL_DISTRIBUTION.value:
            return (
                maximum_likelihood_normal_distribution.
                MaximumLikelihoodNormalDistribution(
                    hypothesis=hypothesis, targets=targets, **kwargs))
        if n == CostFunctionTypes.LOG_LOSS.value:
            return log_loss.LogLoss(
                hypothesis=hypothesis, targets=targets, **kwargs)
        elif n == CostFunctionTypes.PERCEPTRON_BATCH_COST.value:
            return perceptron_batch_cost.PerceptronBatchCost(
                hypothesis=hypothesis, targets=targets, **kwargs)
        elif n == CostFunctionTypes.PERCEPTRON_ONLINE_COST.value:
            return perceptron_online_cost.PerceptronOnlineCost(
                hypothesis=hypothesis, targets=targets, **kwargs)
        else:
            raise ValueError("Invalid cost function id: {}".format(n))

    @staticmethod
    def get_cost_function_by_name(cost_function_name,
                                  hypothesis, targets, **kwargs):
        if cost_function_name == "squared_error_loss":
            return squared_error_loss.SquaredErrorLoss(
                hypothesis=hypothesis, targets=targets, **kwargs)
        if cost_function_name == "maximum_likelihood_normal_distribution":
            return (
                maximum_likelihood_normal_distribution.
                MaximumLikelihoodNormalDistribution(
                    hypothesis=hypothesis, targets=targets, **kwargs))
        if cost_function_name == "log_loss":
            return log_loss.LogLoss(
                hypothesis=hypothesis, targets=targets, **kwargs)
        elif cost_function_name == "perceptron_batch_cost":
            return perceptron_batch_cost.PerceptronBatchCost(
                hypothesis=hypothesis, targets=targets, **kwargs)
        elif cost_function_name == "perceptron_online_cost":
            return perceptron_online_cost.PerceptronOnlineCost(
                hypothesis=hypothesis, targets=targets, **kwargs)
        else:
            raise ValueError(
                "Invalid cost function name: {}".format(cost_function_name))


class AlgorithmTypes(Types):
    BATCH_GRADIENT_DESCENT = 1
    STOCHASTIC_GRADIENT_DESCENT = 2
    NEWTON_RAPHSON = 3


class CostFunctionsAlgorithms(object):
    @staticmethod
    def cost_function_algorithm_dict():
        return {CostFunctionTypes.SQUARED_ERROR_LOSS: set(
            [AlgorithmTypes.BATCH_GRADIENT_DESCENT,
             AlgorithmTypes.STOCHASTIC_GRADIENT_DESCENT]),
                CostFunctionTypes.MAXIMUM_LIKELIHOOD_NORMAL_DISTRIBUTION: set(
                    [AlgorithmTypes.BATCH_GRADIENT_DESCENT,
                     AlgorithmTypes.STOCHASTIC_GRADIENT_DESCENT,
                     AlgorithmTypes.NEWTON_RAPHSON]),
                CostFunctionTypes.LOG_LOSS: set(
                    [AlgorithmTypes.BATCH_GRADIENT_DESCENT,
                     AlgorithmTypes.STOCHASTIC_GRADIENT_DESCENT,
                     AlgorithmTypes.NEWTON_RAPHSON]),
                CostFunctionTypes.PERCEPTRON_BATCH_COST: set(
                    [AlgorithmTypes.BATCH_GRADIENT_DESCENT]),
                CostFunctionTypes.PERCEPTRON_ONLINE_COST: set(
                    [AlgorithmTypes.STOCHASTIC_GRADIENT_DESCENT])
               }


class AlgorithmFactory(object):

    @staticmethod
    def get_algorithm(n, cost_function, **kwargs):
        if n == AlgorithmTypes.BATCH_GRADIENT_DESCENT.value:
            return batch_gradient_descent.BatchGradientDescent(
                cost_function=cost_function, **kwargs)
        elif n == AlgorithmTypes.STOCHASTIC_GRADIENT_DESCENT.value:
            return stochastic_gradient_descent.StochasticGradientDescent(
                cost_function=cost_function, **kwargs)
        elif n == AlgorithmTypes.NEWTON_RAPHSON.value:
            return newton_raphson.NewtonRaphson(
                cost_function=cost_function, **kwargs)
        else:
            raise ValueError("Invalid algorithm id: {}".format(n))

    @staticmethod
    def get_algorithm_by_name(algorithm_name, cost_function, **kwargs):
        if algorithm_name == "batch_gradient_descent":
            return batch_gradient_descent.BatchGradientDescent(
                cost_function=cost_function, **kwargs)
        elif algorithm_name == "stochastic_gradient_descent":
            return stochastic_gradient_descent.StochasticGradientDescent(
                cost_function=cost_function, **kwargs)
        elif algorithm_name == "newton_raphson":
            return newton_raphson.NewtonRaphson(
                cost_function=cost_function, **kwargs)
        else:
            raise ValueError(
                "Invalid algorithm name: {}".format(algorithm_name))
