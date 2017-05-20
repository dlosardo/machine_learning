"""
runmain takes and validates command line arguments and calls run method
"""
from argparse import ArgumentParser, FileType
import sys
import csv
from machine_learning.driver import run

def main(args=None):
    """Main method that parses command line arguments and calls run method"""
    if args is None:
        args = sys.argv[1: ]
    parser = ArgumentParser(description="Run a Machine Learning algorithm using command line arguments")
    parser.add_argument(
        '--input-data-file', type=FileType('r'), required=True,
        dest='input_data_file', help='csv file for input data')
    parser.add_argument(
        '--number-features', type=int, required=True,
        dest='number_features', help='number of features in input dataset')
    parser.add_argument(
        '--number-targets', type=int, required=True,
        dest='number_targets', help='number of targets in input dataset')
    parser.add_argument(
        '--hypothesis-name', type=str, required=True,
        dest='hypothesis_name', help='Can be one of:')
    parser.add_argument(
        '--cost-function-name', type=str, required=True,
        dest='cost_function_name', help='Can be one of:')
    parser.add_argument(
        '--algorithm-name', type=str, required=True,
        dest='algorithm_name', help='Can be one of:')
    parser.add_argument(
        '--tolerance', type=float, required=False,
        dest='tolerance', help='Can be one of:')
    parser.add_argument(
        '--learning-rate', type=float, required=False,
        dest='learning_rate', help='Can be one of:')
    parser.add_argument(
        '--starting-parameter-values-file', type=FileType('r'), required=False,
        dest='starting_parameter_values_file', help='Can be one of:')
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
