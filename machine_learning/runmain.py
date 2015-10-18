"""
runmain takes and validates command line arguments and calls run method
"""
from argparse import ArgumentParser, FileType
import sys
import csv
from driver import run

def main(args=None):
    """Main method that parses command line arguments and calls run method"""
    if args is None:
        args = sys.argv[1: ]
    parser = ArgumentParser(description="Run a Machine Learning algorithm using command line arguments")
    parser.add_argument(
        '--input-data-file', type=FileType('r'), required=True,
        dest='input_data_file', help='csv file for input data')
    options = parser.parse_args(args)
    output = run(
        options.input_data_file)
