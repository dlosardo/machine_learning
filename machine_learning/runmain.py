"""
runmain takes and validates command line arguments and calls run method
"""
from argparse import ArgumentParser, FileType
import sys, csv
from machine_learning.model_utils.factories import HypothesisTypes, CostFunctionTypes, AlgorithmTypes
from machine_learning.driver.command_line_driver import run

hypothesis_choices=[hypothesis_type.name.lower() for hypothesis_type in list(HypothesisTypes)]
cost_function_choices=[cost_function_type.name.lower() for cost_function_type in list(CostFunctionTypes)]
algorithm_choices=[algorithm_type.name.lower() for algorithm_type in list(AlgorithmTypes)]

def main(args=None):
    """Main method that parses command line arguments and calls run method"""
    if args is None:
        args = sys.argv[1: ]
    else:
        print(args)
    parser = ArgumentParser(description="Run a Machine Learning algorithm using command line arguments")
    parser.add_argument(
        '--input-data-file', type=FileType('r'), required=True,
        dest='input_data_file', help='csv file for input data')
    parser.add_argument(
        '--number-features', type=int, required=True,
        dest='number_features', help='An integer representing number of features in input dataset')
    parser.add_argument(
        '--number-targets', type=int, required=False,
        default = 0,
        dest='number_targets', help='An integer representing number of targets in input dataset')
    parser.add_argument(
        '--hypothesis-name', type=str, required=True,
        choices=hypothesis_choices,
        dest='hypothesis_name', help='Can be one of: {}'.format(", ".join(hypothesis_choices)))
    parser.add_argument(
        '--cost-function-name', type=str, required=True,
        choices=cost_function_choices,
        dest='cost_function_name', help='Can be one of: {}'.format(", ".join(cost_function_choices)))
    parser.add_argument(
        '--algorithm-name', type=str, required=True,
        choices=algorithm_choices,
        dest='algorithm_name', help='Can be one of: {}'.format(", ".join(algorithm_choices)))
    parser.add_argument(
        '--tolerance', type=float, required=False,
        default = None,
        dest='tolerance', help='A float value representing the tolerance for stopping criteria of an algorithm')
    parser.add_argument(
        '--learning-rate', type=float, required=False,
        default = None,
        dest='learning_rate', help='A float value representing the learning rate of an algorithm')
    parser.add_argument(
        '--starting-parameter-values-file', type=FileType('r'), required=False,
        default=None,
        dest='starting_parameter_values_file', help='A filename containing starting values for parameters')
    options = parser.parse_args(args)
    output = run(
        options.input_data_file
        , options.number_features
        , options.number_targets
        , options.hypothesis_name
        , options.cost_function_name
        , options.algorithm_name
        , options.learning_rate
        , options.tolerance
        , options.starting_parameter_values_file)
    return output
