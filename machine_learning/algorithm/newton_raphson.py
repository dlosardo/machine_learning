"""
Newton-Raphson Algorithm
IS A SupervisedAlgorithm
Need to know the matrix of second derivatives wrt parameters.
theta_next = theta_current - partial derivative of likelihood wrt theta / partial second
    derivative of likelihood wrt theta
with this method we can derive standard errors!
they are sqrt(1 / negative partial second derivative of likelihood wrt parameters)
"""
from machine_learning.algorithm.iterative_supervised_algorithm import IterativeSupervisedAlgorithm
from numpy.linalg import inv, norm


class NewtonRaphson(IterativeSupervisedAlgorithm):
    def __init__(self, cost_function, tolerance=None, param_starting_values=None):
        """
        Constructor for NewtonRaphson
        :param param_starting_values A dict of the form parameter_name: parameter_value
        :cost_function A CostFunction object, e.g., SquaredErrorLoss
        """
        super(NewtonRaphson, self).__init__(cost_function, param_starting_values)
        if tolerance is None:
            self.tolerance = .001
        else:
            self.tolerance = tolerance
        self.current_theta = None
        self.next_theta = None

    def reset(self):
        """
        Resets the state of the algorithm
        """
        self.current_theta = None
        self.next_theta = None
        self.iter = 0
        self.converged = False
        self.cost_function_list = []
        self.convergence_value_list = []
        self.initialize_parameters()

    def iterate(self):
        """
        Performs one iteration step
        Updates parameters with new values using cost function
        """
        self.current_theta = self.get_parameters()
        self.next_theta = self.current_theta - inv(self.cost_function.cost_function_second_derivative()).dot(self.cost_function.cost_function_derivative())
        self.cost_function.update_parameters(self.next_theta)

    def convergence_criteria_met(self):
        return self.convergence_value() < self.tolerance

    def convergence_value(self):
        return norm(self.next_theta - self.current_theta)
