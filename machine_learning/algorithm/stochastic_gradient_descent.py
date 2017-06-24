"""
"""
from machine_learning.algorithm.gradient_descent import GradientDescent


class StochasticGradientDescent(GradientDescent):
    """Constructor
    """
    def __init__(self, cost_function, learning_rate, tolerance=None, param_starting_values=None):
        super(StochasticGradientDescent, self).__init__(cost_function, learning_rate=learning_rate
                , tolerance=tolerance, param_starting_values=param_starting_values)

    def iterate(self):
        """One iteration step
        1. Set the current cost to the calculation of the cost function using the current parameter estimates
        2. Calculate updated parameter estimates
        3. Update parameter estimates
        4. Compute new cost using new parameter estimates
        """
        self.current_cost = self.cost_function.cost_function()
        for i in range(0, self.nobs):
            updated_params = self.get_parameters() + self.learning_rate*self.cost_function.cost_function_derivative(i)
            self.cost_function.update_parameters(updated_params)
        self.new_cost = self.cost_function.cost_function()

    def shuffle_parameters(self):
        pass
