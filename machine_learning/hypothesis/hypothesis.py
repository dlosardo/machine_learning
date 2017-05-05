"""
Hypothesis
"""
from machine_learning.parameter import ParameterList

class Hypothesis(object):

    def __init__(self, features):
        self.parameter_list = ParameterList()
        self.features = features

    def parameters_initialized(self):
        """Checks to see whether parameters have been initialized.
        :return True if parameters have been initialized, false otherwise
        """
        return self.parameter_list.all_parameters_initialized()

    def initialize_parameters(self, param_dict=None):
        """Initialize the parameter values
        :param param_values A dictionary of parameter_name:parameter_value
        """
        self.parameter_list.initialize_parameters(param_dict)

    def update_parameters(self, param_array):
        """Update parameter values
        :param param_array A numpy array of parameter values of dimension nparms x 1
        """
        raise NotImplementedError

    def get_parameters(self):
        raise NotImplementedError

    def hypothesis_function(self, features):
        raise NotImplementedError
