# command line driver for machine_learning

# Set up

# Hypothesis: hypothesis_function(features)
#         |
#         V
# SimpleLinearRegression -> parameters intercept, slope


# MachineLearningAlgorithm
#         |
#         V
#  SupervisedAlgorithm(CostFunction)
#         |
#         V
#  GradientDescent(learning_rate, CostFunction)


# CostFunction(hypothesis, targets)
#        |
#        V
#  SquaredErrorLoss: methods: cost_function, cost_function_derivative

#mkvirtualenv machine_learning
#python setup.py develop
#run.py --input-data-file "data/input/ex1data1.txt"

import numpy as np
from machine_learning.factories import HypothesisFactory, CostFunctionFactory, AlgorithmFactory
import machine_learning.utils.utils

def get_input_data(input_data_file):
    data = np.genfromtxt(input_data_file, dtype=float, delimiter = ',')
    return data

def extract_data(data, number_features, number_targets):
    # extract features
    features = data[:, 0:number_features]
    # reshape to dimension nobs x nfeatures
    features = features.reshape((len(features), number_features))
    # extract targets
    targets = data[:, number_features:(number_features + number_targets)]
    # reshape to dimension nobs x 1
    targets = targets.reshape((len(targets), number_targets))
    return features, targets


def set_hypothesis(hypothesis_name, features, **kwargs):
    hypothesis_object = HypothesisFactory.get_hypothesis_by_name(hypothesis_name, features, **kwargs)
    return hypothesis_object

def set_cost_function(cost_function_name, hypothesis, targets, **kwargs):
    cost_function_object = CostFunctionFactory.get_cost_function_by_name(
            cost_function_name, hypothesis, targets, **kwargs)
    return cost_function_object

def set_algorithm(algorithm_name, cost_function, starting_parameter_values, **kwargs):
    algorithm_object = AlgorithmFactory.get_algorithm_by_name(algorithm_name
            , cost_function=cost_function, starting_parameter_values=starting_parameter_values
            , **kwargs)
    return algorithm_object

def run(input_data_file, number_features, number_targets, hypothesis_name, cost_function_name, algorithm_name
        , learning_rate, tolerance, starting_parameter_values_file):
    """
    """
    if starting_parameter_values_file is None:
        starting_parameter_values = None
    #TODO: parse file to dict
    # read in data
    data = get_input_data(input_data_file)
    features, targets = extract_data(data, number_features, number_targets)
    # set hypothesis, cost function, and algorithm
    hypo = set_hypothesis(hypothesis_name, features)
    cost_fnx = set_cost_function(cost_function_name, hypo, targets)
    optional_arguments = [{"learning_rate": learning_rate}, {"tolerance": tolerance}]
    algo_kwargs = {}
    for optional_argument in optional_arguments:
        for key, val in optional_argument.items():
            if val is not None:
                algo_kwargs.update(optional_argument)

    algo = set_algorithm(algorithm_name, cost_fnx, starting_parameter_values, **algo_kwargs)
    # run algorithm
    algo.algorithm()
    print(algo.get_parameters())
