"""
Gradient Descent ML Algorithm
IS A SupervisedAlgorithm
"""
from machine_learning.algorithm.supervised_algorithm import SupervisedAlgorithm


class GradientDescent(SupervisedAlgorithm):
    """Constructor for GradientDescent
    :param learning_rate A float value representing the learning rate
    :param param_starting_values A dict of the form parameter_name: parameter_value
    :tolerance A float value representing the tolerance value used to inform convergence specifications
    :cost_function A CostFunction object, e.g., SquaredErrorLoss
    """
    def __init__(self, cost_function, learning_rate, tolerance=None, param_starting_values=None):
        super(GradientDescent, self).__init__()
        self.learning_rate = learning_rate
        self.cost_function = cost_function
        self.tolerance = tolerance
        self.param_starting_values = param_starting_values
        self.nobs = self.cost_function.nobs
        self.current_cost = None
        self.new_cost = None
        self.iter = 0

    def initialize_parameters(self):
        """Initialize parameters
        """
        self.cost_function.initialize_parameters(self.param_starting_values)

    def get_parameters(self):
        """Return parameters
        """
        return self.cost_function.get_parameters()

    def iterate(self):
        raise NotImplementedError

    def algorithm(self):
        """Run the algorithm
        Call iterate until the change in the cost function is less than the specified tolerance value
        """
        self.initialize_parameters()
        while (True):
            self.iterate()
            if self.iter % 100000 == 0:
                print("iter: {}".format(self.iter))
            if (self.cost_function.convergence_criteria_met(self.current_cost, self.new_cost, self.tolerance)):
                break
            self.iter = self.iter + 1
