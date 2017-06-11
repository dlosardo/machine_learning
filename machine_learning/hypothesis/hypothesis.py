"""
Hypothesis
The hypothesis is a function of the inputs (aka: x, features)
"""
from machine_learning.model_utils.parameter import ParameterList
from machine_learning.utils.exceptions import ParameterValuesNotInitialized, IncorrectMatrixDimensions

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
        if not self.parameters_initialized():
            raise ParameterValuesNotInitialized(
                    "Parameter values have not yet been initialized")
        if (param_array.shape[0] != self.nparams | len(param_array.shape) != 2 | param_array[1] != 1):
            raise IncorrectMatrixDimensions(
                    "Parameter array needs to be %d by 1" % self.nparams)
        for i, param in enumerate(self.parameter_list.parameter_list):
            param.value = param_array[i][0]

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for later computation.
        First horizontally stacks all parameter values
        Next reshapes into an array of dimension number of parameters by 1.
        :returns A numpy array of dimension number of nparams by 1
         containing the values of the parameters.
        """
        if not self.parameters_initialized():
            raise ParameterValuesNotInitialized(
                    "Parameter values have not yet been initialized")
        return self.parameter_list.get_parameters()

    def hypothesis_function(self):
        raise NotImplementedError
