"""
Utility functions for mathematical operations
"""
from numpy import append, ones


def add_constant(numpy_array):
    numpy_array_with_constant = append(ones(numpy_array.shape[0]).reshape(
        numpy_array.shape[0], 1), numpy_array, 1)
    return numpy_array_with_constant
