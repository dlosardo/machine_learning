# driver for machine_learning

# Set up

# Hypothesis: methods: get_parameters, hypothesis_function(features)
#         |
#         V
# SimpleLinearRegression(intercept, slope)


# MachineLearningAlgorithm
#         |
#         V
#  SupervisedAlgorithm(features, yvalues)
#         |
#         V
#  GradientDescent(learning_rate, CostFunction, Hypothesis)
#                 methods: parameter_derivatives, iterate


# CostFunction(hypothesis, features, yvalues)
#        |
#        V
#  SquaredErrorLoss: methods: cost_function, cost_function_derivative

#mkvirtualenv machine_learning
#python setup.py develop
#run.py --input-data-file "data/input/ex1data1.txt"

import numpy as np
from simple_linear_regression import SimpleLinearRegression
from squared_error_loss import SquaredErrorLoss
from gradient_descent import GradientDescent

def get_input_data(input_data_file):
    data = np.genfromtxt(input_data_file, dtype=float, delimiter = ',')
    return data

#def extract_data()

def run(
        input_data_file):
    """
    """
    # parameter values
    intercept = 0.
    slope = 0.
    param_values = {"intercept": intercept,
            "slope": slope}
    # hypothesis object
    hypo = SimpleLinearRegression()
    hypo.initialize_parameters(param_values)

    # name of file where data is stored
    #file_name = 'data/input/ex1data1.txt'
    # read in data
    data = get_input_data(input_data_file)
    # extract features
    features = data[:, 0]
    # reshape to dimension nobs x 1
    features = features.reshape((len(features), 1))
    # extract yvalues
    yvalues = data[:, 1]
    # reshape to dimension nobs x 1
    yvalues = yvalues.reshape((len(yvalues), 1))

    # cost function object
    sel = SquaredErrorLoss(hypo, features, yvalues)
    gd = GradientDescent(.0001, param_values, .000000001, sel)
    gd.algorithm()
    print gd.get_parameters()
