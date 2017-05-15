"""
Perceptron Learning ML Algorithm
IS A SupervisedAlgorithm
"""
from machine_learning.algorithm.supervised_algorithm import SupervisedAlgorithm


class StochasticGradientDescent(SupervisedAlgorithm):
    """Constructor for PerceptronLearning
    :param learning_rate A float value representing the learning rate
    :cost_function A CostFunction object, e.g., SquaredErrorLoss
    :param param_starting_values A dict of the form parameter_name: parameter_value
    """
    def __init__(self, learning_rate, cost_function, tolerance=None, param_starting_values=None):
        super(StochasticGradientDescent, self).__init__()
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
        """One iteration step
        1. Set the current cost to the calculation of the cost function using the current parameter estimates
        2. Calculate updated parameter estimates
        3. Update parameter estimates
        4. Compute new cost using new parameter estimates
        """
        self.current_cost = self.cost_function.cost_function()
        print(self.get_parameters())
        for i in range(0, self.nobs):
            updated_params = self.get_parameters() + self.learning_rate*self.cost_function.cost_function_derivative(i)
            self.cost_function.update_parameters(updated_params)
        print(self.get_parameters())
        self.new_cost = self.cost_function.cost_function()

    def algorithm(self):
        """Run the algorithm
        Call iterate until the change in the cost function is less than the specified tolerance value
        """
        self.initialize_parameters()
        while (True):
            self.iterate()
            if self.iter % 100000 == 0:
                print("iter: {}".format(self.iter))
                #print("number errors: {}".format(self.number_errors))
            if (self.cost_function.convergence_criteria_met(self.current_cost, self.new_cost, self.tolerance)):
                break
            self.iter = self.iter + 1

