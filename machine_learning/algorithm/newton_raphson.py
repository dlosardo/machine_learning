"""
Newton-Raphson Algorithm
IS A SupervisedAlgorithm
Need to know the matrix of second derivatives wrt parameters.
theta_next = theta_current - partial derivative of likelihood wrt theta / partial second
    derivative of likelihood wrt theta
with this method we can derive standard errors!
they are sqrt(1 / negative partial second derivative of likelihood wrt parameters)
"""
from machine_learning.algorithm.supervised_algorithm import SupervisedAlgorithm
from numpy.linalg import inv, norm
from numpy import sqrt, diag

class NewtonRaphson(SupervisedAlgorithm):
    """Constructor for NewtonRaphson
    :param param_starting_values A dict of the form parameter_name: parameter_value
    :cost_function A CostFunction object, e.g., SquaredErrorLoss
    """
    def __init__(self, cost_function, tolerance=None, param_starting_values=None):
        super(NewtonRaphson, self).__init__(cost_function, param_starting_values)
        if tolerance is None:
            self.tolerance = .001
        else:
            self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.current_theta = None
        self.next_theta = None
        self.iter = 0
        self.converged = False

    def iterate(self):
        self.current_theta = self.get_parameters()
        self.next_theta = self.current_theta + inv(self.cost_function.cost_function_second_derivative()).dot(self.cost_function.cost_function_derivative())
        self.cost_function.update_parameters(self.next_theta)

    def algorithm(self):
        """Run the algorithm
        """
        self.initialize_parameters()
        self.current_theta = self.get_parameters()
        while (True):
            self.iterate()
            if self.iter % 100000 == 0:
                print("iter: {}".format(self.iter))
            if (norm(self.next_theta - self.current_theta) < self.tolerance):
                self.converged = True
                break
            self.iter = self.iter + 1

    def variance_covariance_matrix(self):
        if self.converged:
            return -1.*inv(self.cost_function.cost_function_second_derivative())
        else:
            raise Exception("Model has not converged")

    def standard_errors(self):
        return sqrt(diag(self.variance_covariance_matrix()))/sqrt(self.nobs)
