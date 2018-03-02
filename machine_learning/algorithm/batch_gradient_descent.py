"""
Batch Gradient Descent ML Algorithm
IS A SupervisedAlgorithm and GradientDescent
"""
from machine_learning.algorithm.gradient_descent import GradientDescent


class BatchGradientDescent(GradientDescent):
    def __init__(self, cost_function, learning_rate, tolerance=None, param_starting_values=None):
        """
        Constructor for GradientDescent
        :param cost_function: A valid CostFunction object
        :param learning_rate: A float representing the learning rate in the algorithm
        :param tolerance: A float representing the degree of tolerance for convergence
        :param param_starting_values: A dictionary of starting values for parameters
        """
        super(BatchGradientDescent, self).__init__(cost_function, learning_rate=learning_rate
                , tolerance=tolerance, param_starting_values=param_starting_values)

    def iterate(self):
        """
        One iteration step
        1. Set the current cost to the calculation of the cost function using the current parameter estimates
        2. Calculate updated parameter estimates
        3. Update parameter estimates
        4. Compute new cost using new parameter estimates
        """
        self.current_cost = self.cost_function.cost_function()

        updated_params = self.get_parameters() - self.learning_rate*self.cost_function.cost_function_derivative()
        self.cost_function.update_parameters(updated_params)
        self.new_cost = self.cost_function.cost_function()
