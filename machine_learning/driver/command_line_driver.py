"""
command line driver for machine_learning
General Set Up:
 Hypothesis: hypothesis_function(features)
         |
         V
 SimpleLinearRegression -> parameters intercept, slope

 MachineLearningAlgorithm
         |
         V
  SupervisedAlgorithm(CostFunction)
         |
         V
  GradientDescent(learning_rate, CostFunction)

 CostFunction(hypothesis, targets)
        |
        V
  SquaredErrorLoss: methods: cost_function, cost_function_derivative
"""
import numpy as np
import logging
from machine_learning.model_utils.model_setup import ModelSetup
from machine_learning.model_utils.learning_model import LearningModel
import machine_learning.utils.utils

def get_input_data(input_data_file):
    data = np.genfromtxt(input_data_file, dtype=float, delimiter = ',')
    return data

def extract_data(data, number_features, number_targets):
    # extract features
    features = data[:, 0:number_features]
    # reshape to dimension nobs x nfeatures
    features = features.reshape((len(features), number_features))
    if number_targets == 0:
        targets = None
    else:
        # extract targets
        targets = data[:, number_features:(number_features + number_targets)]
        # reshape to dimension nobs x 1
        targets = targets.reshape((len(targets), number_targets))
    return features, targets

def run(input_data_file, number_features, number_targets, hypothesis_name, cost_function_name, algorithm_name
        , learning_rate, tolerance, starting_parameter_values_file):
    """
    """
    if starting_parameter_values_file is None:
        starting_parameter_values = None
    #TODO: parse file to dict
    logging.info(input_data_file)
    # setup model and check model dependencies (will fail loudly here if not set up properly)
    model_setup_obj = ModelSetup(hypothesis_name, cost_function_name, algorithm_name)
    # read in data
    data = get_input_data(input_data_file)
    features, targets = extract_data(data, number_features, number_targets)
    # obtain algorithm object
    algorithm_obj = model_setup_obj.model_setup(features, targets, learning_rate
     , tolerance, starting_parameter_values)
    # create learning model object and run model and print results
    learning_model_obj = LearningModel(algorithm_obj)
    learning_model_obj.run_model()
    learning_model_obj.print_results()
    return learning_model_obj
