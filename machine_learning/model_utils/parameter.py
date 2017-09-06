"""
Parameter Class
"""
from numpy import hstack, diag, sqrt, ndarray
import sympy as sp
from tabulate import tabulate
import itertools
from machine_learning.utils.exceptions import ParameterValuesNotInitialized, IncorrectMatrixDimensions


class Parameter(object):
    def __init__(self, name, value, variance, default_starting_value):
        """
        Constructor
        Parameter has a name, a value, a variance, and a default starting value.
        Name must be a string.
        Value must be None or a float.
        Variance must be None or a float.
        Default Starting Value must be a float.
        """
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

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name

    def is_initialized(self):
        return self.value is not None


"""
ParameterList Class
"""
class ParameterList(object):
    def __init__(self):
        """
        Constructor
        Initialized to an empty list
        """
        self.size = 0
        self.__parameter_list = []
        self.parameter_covariance_matrix = None
        self.symbolic_covariance_matrix = None

    @property
    def parameter_list(self):
        """
        Don't want to be able to set parameter_list again
        """
        return self.__parameter_list

    def add_parameter(self, parameter_object):
        """
        Adds a parameter object to the end of the list
        If parameter object is already in the list, throws exception
        :param parameter_object: An object of type Parameter
        """
        if not isinstance(parameter_object, Parameter):
            raise TypeError("Must be a Parameter object")
        elif self.contains_parameter(parameter_object):
            raise Exception("Parameter {} already in list, cannot add again".format(parameter_object))
        else:
            self.size = self.size + 1
            self.parameter_list.append(parameter_object)

    def initialize_parameters(self, param_dict=None):
        """
        Initializes all parameters.
        If a param_dict is not supplied, uses default_starting_value of
        each Parameter in the list.
        If a param_dict does not contain the parameter, uses the
        default_starting_value of corresponding Parameter
        :param param_dict: A python dictionary in the form {param_name: param_value}
        """
        if self.size == 0:
            return
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

    def all_parameters_initialized(self):
        """
        Checks whether all of the parameters in the parameter_list are initialized
        :returns: True if all parameters are initialized, False otherwise
        """
        if self.size == 0:
            return False
        for param in self.parameter_list:
            if not param.is_initialized():
                return False
        return True

    def set_parameter_variances(self, param_variance_list):
        """
        Given a list of parameter variances, sets the variance values on each Parameter
        :param param_variance_list: A list of float values equal to the size of parameter_list
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if not isinstance(param_variance_list, list):
            raise TypeError("param_variance_list must be a list")
        if len(param_variance_list) != self.size:
            raise TypeError("param_variance_list must be length {}, found length {}".format(self.size
                , len(param_variance_list)))
        for i, param in enumerate(self.parameter_list):
            param.variance = param_variance_list[i]

    def set_covariance_matrix(self, cov_matrix):
        #TODO: check to see if the variance value is the same, if not throw exception
        #TODO: make methods for getting value and variance by name and covariance by pair of names
        """
        Sets the covariance matrix of the Parameters in parameter_list
        """
        if self.size == 0:
            return
        if not isinstance(cov_matrix, ndarray):
            raise TypeError("cov_matrix must be a Numpy Array object")
        if cov_matrix.ndim != 2:
            raise IncorrectMatrixDimensions("cov_matrix must be two dimensions, found {} dimensions".format(cov_matrix.ndim))
        if not (cov_matrix.shape[0] == self.size and cov_matrix.shape[1] == self.size):
            raise IncorrectMatrixDimensions("cov_matrix must be {} by {}, found {} by {}".format(self.size
                , self.size, cov_matrix.shape[0], cov_matrix.shape[1]))
        self.parameter_covariance_matrix = cov_matrix

    def update_parameters(self, param_array):
        """
        Updates the values of each parameter from the given param_array
        :param param_array: A Numpy Array object of dimension size of parameter_list by 1
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if not self.all_parameters_initialized():
            raise ParameterValuesNotInitialized(
                    "Parameter values have not yet been initialized")
        if not isinstance(param_array, ndarray):
            raise TypeError("param_array must be a Numpy Array object")
        if param_array.ndim != 2:
            raise TypeError("param_array must be two dimensions, found {} dimensions".format(param_array.ndim))
        if not (param_array.shape[0] == self.size and param_array.shape[1] == 1):
            raise TypeError("param_array must be of dimension {} by {}, found {} by {}".format(
                self.size, 1, param_array.shape[0], param_array.shape[1]))
        for i, param in enumerate(self.parameter_list):
            param.value = param_array[i][0]

    def get_parameters(self):
        """
        Reshapes parameters into form suitable for computation.
        First horizontally stacks all parameter values
        Next reshapes into an array of dimension number of parameters by 1.
        :returns: A numpy array of dimension number of params by 1
         containing the values of the parameters.
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if not self.all_parameters_initialized():
            raise ParameterValuesNotInitialized("Not all parameters are initialized")
        return hstack((param.value for param in self.parameter_list)).reshape(self.size, 1)

    def get_parameter_by_name(self, parameter_name):
        """
        Given a parameter name, returns the parameter object
        :param parameter_name: A string representing the name of a parameter
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        for param in self.parameter_list:
            if parameter_name == param.name:
                return param

    def get_parameter_at_index(self, index):
        """
        Retrieves the parameter object at the given index
        :param index: An integer representing the index of a desired parameter
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if (index < 0) or (index >= self.size):
            raise IndexError("index out of bounds, must be between 0 and {}".format(self.size - 1))
        return self.parameter_list[index]

    def get_parameter_names(self):
        """
        Returns a list of parameter names (strings)
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        return [param.name for param in self.parameter_list]

    def get_parameter_values_by_name(self, name_list):
        """
        Given a list of parameter names, returns parameter values
        :param name_list: A list of parameter names
        :returns: A numpy array of parameter values for those in the name_list
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if not self.all_parameters_initialized():
            raise Exception("Not all parameters are initialized")
        parameters_tmp = [self.get_parameter_by_name(name)
                            for name in name_list
                            if self.contains_parameter_by_name(name)]
        if len(parameters_tmp) == 0:
            raise Exception("No parameter names in parameter list")
        return hstack((param.value for param in parameters_tmp)).reshape(len(parameters_tmp), 1)

    def get_parameter_values_not_in_list(self, param_list):
        """
        Given a list of parameter objects, returns parameter values for Parameters found in
        parameter_list but NOT found in param_list
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if not self.all_parameters_initialized():
            raise Exception("Not all parameters are initialized")
        parameters_tmp = [param for param in self.parameter_list if param not in param_list]
        if len(parameters_tmp) == 0:
            raise Exception("All parameters in parameter list")
        return hstack((param.value for param in parameters_tmp)).reshape(len(parameters_tmp), 1)

    def get_parameter_variances(self):
        """
        :returns A numpy array of dimension number of nparams by 1
         containing the variances of the parameters.
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        return hstack((param.variance for param in self.parameter_list)).reshape(self.size, 1)

    def get_parameter_standard_errors(self):
        """
        :returns A numpy array of dimension number of nparams by 1
         containing the standard errors of the parameters.
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        return hstack((sqrt(param.variance) if param.variance is not None else None for param in self.parameter_list)).reshape(self.size, 1)

    def contains_parameter(self, parameter_object):
        """
        Checks whether the parameter_list contains a parameter object
        :param parameter_object: An object of type Parameter
        :returns: True if parameter_list contains the Parameter object,
            False otherwise
        """
        if self.size == 0:
            return False
        if not isinstance(parameter_object, Parameter):
            raise TypeError("Must be a Parameter object")
        for parameter in self.parameter_list:
            if parameter == parameter_object:
                return True
        return False

    def contains_parameter_by_name(self, parameter_name):
        """
        Checks whether the parameter_list contains a parameter object by
        looking for its name
        :param parameter_name: A string representing a parameter name
        :returns: True if parameter_list contains the Parameter object associated
            with the name, False otherwise
        """
        if self.size == 0:
            return False
        if not isinstance(parameter_name, str):
            raise TypeError("Must be string")
        for parameter in self.parameter_list:
            if parameter.name == parameter_name:
                return True
        return False

    def parameter_index(self, parameter_object):
        """
        Given a parameter object, obtains the index it is located at in parameter_list
        :param parameter_object: An object of type Parameter
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        if not isinstance(parameter_object, Parameter):
            raise TypeError("Must be a Parameter object")
        index=0
        for parameter_ in self.parameter_list:
            if parameter_ == parameter_object:
                return index
            index = index + 1
        return -1

    def remove_parameter_from_parameter_object(self, parameter_object):
        """
        Removes the given parameter from parameter_list
        :param parameter_object: An object of type Parameter
        """
        if self.size==0:
            raise Exception("Parameter list is empty, nothing to remove")
        if not isinstance(parameter_object, Parameter):
            raise TypeError("Must be a Parameter object")
        if self.contains_parameter(parameter_object):
            self.parameter_list.remove(parameter_object)
            self.size = self.size - 1
        else:
            raise Exception("Parameter {} not found".format(parameter_object))

    def clear_parameter_list(self):
        """
        Empties the parameter_list
        """
        if self.size == 0:
            raise Exception("Parameter List is empty")
        param_names = self.get_parameter_names()
        for param_name in param_names:
            self.remove_parameter_from_parameter_object(self.get_parameter_by_name(param_name))

    # def parameter_covariance_matrix(self):
        # return diag([param.variance for param in self.parameter_list])

    def get_unique_elements_in_covariance_matrix(self):
        """
        Number of unique elements in the covariance matrix as a function
        of number of parameters in self.parameter_list
        """
        return ((self.size) * (self.size + 1)) / 2

    def set_parameter_covariance_matrix_symbolic(self):
        """
        Creates a symbolic covariance matrix representing all pairs of covariances
        of parameter in the off-diagonal and parameter variances in the diagonal
        """
        if self.size > 0:
            covariance_pairs = self.get_covariance_pairs()
            self.symbolic_covariance_matrix = sp.Matrix(self.size, self.size, lambda i,j: sp.var(
                                             'cov({}, {})'.format(covariance_pairs[i + j + (j*(self.size-1))][0],
                                             covariance_pairs[i + j +(j*(self.size-1))][1])))

    def get_covariance_pairs(self):
        """
        Obtains all pairs of parameters
        """
        if self.size > 0:
            covariance_pairs = [i for i in itertools.product(self.get_parameter_names(), repeat=2)]
            return covariance_pairs

    def print_parameters(self):
        """
        Prints the parameter names, point estimates, and standard errors
        """
        names_values = list(zip(self.get_parameter_names(), list(self.get_parameters().ravel())
            , list(self.get_parameter_standard_errors().ravel())))
        print(tabulate(names_values, headers=['Parameter', 'Point Estimate', 'Standard Errors']))
