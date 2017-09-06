"""
runmain takes and validates command line arguments and calls run method
"""
from argparse import ArgumentParser, FileType
import sys
from machine_learning.model_utils.factories import HypothesisTypes, CostFunctionTypes, AlgorithmTypes
from machine_learning.driver.command_line_driver import run

def main(args=None):
    """
    Main method that parses command line arguments and calls run method
    """
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
        choices=HypothesisTypes.names_list(),
        dest='hypothesis_name', help='Can be one of: {}'.format(", ".join(HypothesisTypes.names_list())))
    parser.add_argument(
        '--cost-function-name', type=str, required=True,
        choices=CostFunctionTypes.names_list(),
        dest='cost_function_name', help='Can be one of: {}'.format(", ".join(CostFunctionTypes.names_list())))
    parser.add_argument(
        '--algorithm-name', type=str, required=True,
        choices=AlgorithmTypes.names_list(),
        dest='algorithm_name', help='Can be one of: {}'.format(", ".join(AlgorithmTypes.names_list())))
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
        , HypothesisTypes[options.hypothesis_name.upper()]
        , CostFunctionTypes[options.cost_function_name.upper()]
        , AlgorithmTypes[options.algorithm_name.upper()]
        , options.learning_rate
        , options.tolerance
        , options.starting_parameter_values_file)
    return output
