"""
Parameter Class
"""
from numpy import hstack, diag, sqrt
import sympy as sp
from tabulate import tabulate
import itertools

class Parameter(object):
    """
    Constructor
    Parameter has a name, a value, and a default starting value.
    Name must be a string.
    Value must be None or a float.
    """
    def __init__(self, name, value, variance, default_starting_value):
        self.name = name
        self.value = value
        self.variance = variance
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

    @property
    def variance(self):
        return self.__variance

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

    @variance.setter
    def variance(self, variance):
        if not isinstance(variance, float) and variance is not None:
            raise TypeError("Parameter variance must be None or a float")
        self.__variance = variance

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
        self.parameter_covariance_matrix = None
        self.symbolic_covariance_matrix = None

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

    def set_parameter_variances(self, param_variance_list):
        for i, param in enumerate(self.parameter_list):
            param.variance = param_variance_list[i]
        # if param_variance_dict is None:
            # return
        # else:
            # for param in self.parameter_list:
                # for name, value in param_variance_dict.items():
                    # if (param.name == name) & (value is not None):
                        # param.variance = value

    def set_covariance_matrix(self, cov_matrix):
        self.parameter_covariance_matrix = cov_matrix

    def update_parameters(self, param_array):
        for i, param in enumerate(self.parameter_list):
            param.value = param_array[i][0]
        #for param in self.parameter_list:
         #   for name, value in param_dict.items():
          #      if (param.name == name):
           #         param.value = value

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for computation.
        First horizontally stacks all parameter values
        Next reshapes into an array of dimension number of parameters by 1.
        :returns A numpy array of dimension number of nparams by 1
         containing the values of the parameters.
        """
        return hstack((param.value for param in self.parameter_list)).reshape(self.size, 1)

    def print_parameters(self):
        names_values = list(zip(self.get_parameter_names(), list(self.get_parameters().ravel())
            , list(self.get_parameter_standard_errors().ravel())))
        print(tabulate(names_values, headers=['Parameter', 'Point Estimate', 'Standard Errors']))

    def get_parameter_variances(self):
        """
        :returns A numpy array of dimension number of nparams by 1
         containing the variances of the parameters.
        """
        return hstack((param.variance for param in self.parameter_list)).reshape(self.size, 1)

    def get_parameter_standard_errors(self):
        """
        :returns A numpy array of dimension number of nparams by 1
         containing the standard errors of the parameters.
        """
        return hstack((sqrt(param.variance) if param.variance is not None else None for param in self.parameter_list)).reshape(self.size, 1)

    def get_parameter_names(self):
        return [param.name for param in self.parameter_list]

    def get_parameter_values_by_name(self, name_list):
        parameters_tmp = [self.get_parameter_by_name(name) for name in name_list]
        return hstack((param.value for param in parameters_tmp)).reshape(len(parameters_tmp), 1)

    def parameter_covariance_matrix(self):
        return diag([param.variance for param in self.parameter_list])

    def set_parameter_covariance_matrix_symbolic(self):
        if self.size > 0:
            covariance_pairs = [i for i in itertools.product(self.get_parameter_names(), repeat=2)]
            self.symbolic_covariance_matrix = sp.Matrix(self.size, self.size, lambda i,j: sp.var(
                                             'cov({}, {})'.format(covariance_pairs[i + j + (j*(self.size-1))][0],
                                             covariance_pairs[i + j +(j*(self.size-1))][1])))

    def covariance_pairs(self):
        covariance_pairs = [i for i in itertools.product(self.get_parameter_names(), repeat=2)]
        return covariance_pairs
