"""
Exceptions
"""


class IncorrectMatrixDimensions(Exception):
    """
    Exception when incorrect matrix dimensions are found
    """


class ParameterValuesNotInitialized(Exception):
    """
    Exception when parameter values are not initialized
    """


class HypothesisCostFunctionDependencyException(Exception):
    """
    Exception when you cannot use the given hypothesis with
    the given cost function
    """
    def __init__(self, hypothesis_name, cost_function_name, **kwargs):
        self.message = "Hypothesis type {} is not valid with Cost Function type {}".format(hypothesis_name
                , cost_function_name)
        if kwargs is not '{}':
            for i, v in kwargs.items():
                added_message = '\n' + str(v.__class__) + ': ' + v.message
                self.message += added_message
        super(HypothesisCostFunctionDependencyException, self).__init__(self.message)


class CostFunctionAlgorithmDependencyException(Exception):
    """
    Exception when you cannot use the given cost function with
    the given algorithm
    """
    def __init__(self, cost_function_name, algorithm_name, **kwargs):
        self.message = "Cost function type {} is not valid with Algorithm type {}".format(cost_function_name,
                algorithm_name)
        if kwargs is not '{}':
            for i, v in kwargs.items():
                added_message = '\n' + str(v.__class__) + ': ' + v.message
                self.message += added_message
        super(CostFunctionAlgorithmDependencyException, self).__init__(self.message)
