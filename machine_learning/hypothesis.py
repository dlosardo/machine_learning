"""
Hypothesis
"""

class Hypothesis(object):

    def __init__(self):
        pass

    def initialize_parameters(self, param_dict):
        """Initialize the parameter values
        :param param_values A dictionary of parameter_name:parameter_value
        """
        raise NotImplementedError

    def update_parameters(self, param_array):
        """Update parameter values
        :param param_array A numpy array of parameter values of dimension nparms x 1
        """
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def hypothesis_function(self, features):
        raise NotImplementedError
