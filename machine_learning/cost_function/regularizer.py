"""
Regularizer for cost function
A penalty term for the size of the weights.
None
LASSO
Ridge
Elastic net
L0
L1
L2
"""


class Regularizer(object):

    def lasso_function(self):
        pass

    @classmethod
    def ridge_function(cls, regularization_matrix, parameter_vector):
        return regularization_matrix.dot(
            parameter_vector).T.dot(parameter_vector)

    @classmethod
    def ridge_derivative(cls, regularization_matrix, parameter_vector):
        return regularization_matrix.dot(parameter_vector)

    @classmethod
    def ridge_second_derivative(cls, regularization_matrix, parameter_vector):
        return regularization_matrix

    @classmethod
    def elastic_net_function(self):
        pass

    @classmethod
    def regularizer_cost_function(cls, name, regularization_matrix,
                                  parameter_vector):
        if name == "ridge":
            return cls.ridge_function(regularization_matrix, parameter_vector)
        else:
            return 0

    @classmethod
    def regularizer_cost_function_derivative(cls, name,
                                             regularization_matrix,
                                             parameter_vector):
        if name == "ridge":
            return cls.ridge_derivative(regularization_matrix,
                                        parameter_vector)
        else:
            return 0

    @classmethod
    def regularizer_cost_function_second_derivative(cls, name,
                                                    regularization_matrix,
                                                    parameter_vector):
        if name == "ridge":
            return cls.ridge_second_derivative(regularization_matrix,
                                               parameter_vector)
        else:
            return 0
