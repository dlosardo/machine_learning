"""
Hypothesis
The hypothesis is a function of the inputs (aka: x, features)
"""
from machine_learning.model_utils.parameter import ParameterList


class Hypothesis(object):
    def __init__(self, features):
        """
        Constructor for Hypothesis
        Creates a ParameterList object
        :param features: A nobs by 1 numpy array of feature variables
        """
        self.parameter_list = ParameterList()
        self.features = features
        self.nfeatures = self.features.shape[1]
        self.nobs = self.features.shape[0]
        self.nparams = 0

    def parameters_initialized(self):
        """
        Checks if parameters have been initialized.
        :returns: True if parameters have been initialized, false otherwise
        """
        return self.parameter_list.all_parameters_initialized()

    def initialize_parameters(self, param_dict=None):
        """
        Initialize the parameter values
        :param param_values: A dictionary of {parameter_name: parameter_value}
        """
        self.parameter_list.initialize_parameters(param_dict)

    def update_parameters(self, param_array):
        """
        Update parameter values
        :param param_array: A numpy array of parameter
            values of dimension nparms x 1
        """
        self.parameter_list.update_parameters(param_array)

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for later computation.
        First horizontally stacks all parameter values
        Next reshapes into an array of dimension number of parameters by 1.
        :returns: A numpy array of dimension number of nparams by 1
         containing the values of the parameters.
        """
        return self.parameter_list.get_parameters()

    def reset(self):
        """
        Resets the parameters.
        First clears the list and then sets them to their starting state
        """
        self.parameter_list.clear_parameter_list()
        self.nparams = 0
        self.set_parameters()

    def set_parameters(self):
        """
        Abstract method for setting parameters.
        Depends on the specific Hypothesis to properly set the parameters
        """
        raise NotImplementedError

    def hypothesis_function(self):
        """
        Abstract method to compute the hypothesis function
        :returns: A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        raise NotImplementedError
