"""
Parameter Class
"""
from numpy import hstack

class Parameter(object):
    """
    Constructor
    Parameter has a name, a value, and a default starting value.
    Name must be a string.
    Value must be None or a float.
    """
    def __init__(self, name, value, default_starting_value):
        self.name = name
        self.value = value
        self.default_starting_value = default_starting_value

    @property
    def name(self):
        return self.__name

    @property
    def value(self):
        return self.__value

    @property
    def default_starting_value(self):
        return self.__default_starting_value

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise TypeError("Parameter name must be a string")
        self.__name = name

    @value.setter
    def value(self, value):
        if not isinstance(value, float) and value is not None:
            raise TypeError("Parameter value must be None or a float")
        self.__value = value

    @default_starting_value.setter
    def default_starting_value(self, default_starting_value):
        if not isinstance(default_starting_value, float):
            raise TypeError("Parameter default starting value must be a float")
        self.__default_starting_value = default_starting_value

    def is_initialized(self):
        return self.value is not None


class ParameterList(object):
    """
    Constructor
    Initialized to an empty list
    """
    def __init__(self):
        self.size = 0
        self.parameter_list = []

    def add_parameter(self, parameter_object):
        if not isinstance(parameter_object, Parameter):
            raise TypeError("Must be a Parameter object")
        else:
            self.size = self.size + 1
            self.parameter_list.append(parameter_object)

    def get_parameter_by_name(self, parameter_name):
        for param in self.parameter_list:
            if parameter_name == param.name:
                return param

    def all_parameters_initialized(self):
        if self.size == 0:
            return False
        for param in self.parameter_list:
            if not param.is_initialized():
                return False
        return True

    def initialize_parameters(self, param_dict=None):
        if param_dict is None:
            for param in self.parameter_list:
                param.value = param.default_starting_value
        else:
            for param in self.parameter_list:
                for name, value in param_dict.items():
                    if (param.name == name) & (value is not None):
                        param.value = value
                if not param.is_initialized():
                    param.value = param.default_starting_value

    def update_parameters(self, param_dict):
        for param in self.parameter_list:
            for name, value in param_dict.items():
                if (param.name == name):
                    param.value = value

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for later computation.
        First horizontally stacks all parameter values
        Next reshapes into an array of dimension number of parameters by 1.
        :returns A numpy array of dimension number of nparams by 1
         containing the values of the parameters.
        """
        return hstack((param.value for param in self.parameter_list)).reshape(self.size, 1)
