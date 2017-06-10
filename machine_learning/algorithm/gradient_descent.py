"""
Gradient Descent ML Algorithm
IS A SupervisedAlgorithm
"""
from machine_learning.algorithm.supervised_algorithm import SupervisedAlgorithm


class GradientDescent(SupervisedAlgorithm):
    """Constructor for GradientDescent
    :param learning_rate A float value representing the learning rate
    :param tolerance A float value representing the tolerance value used to inform convergence specifications
    """
    def __init__(self, cost_function, learning_rate=None, tolerance=None, param_starting_values=None):
        super(GradientDescent, self).__init__(cost_function, param_starting_values)
        if learning_rate is None:
            self.learning_rate = 1.
        else:
            self.learning_rate = learning_rate
        if tolerance is None:
            self.tolerance = .001
        else:
            self.tolerance = tolerance
        self.reset()

    def reset(self):
        self.current_cost = None
        self.new_cost = None
        self.iter = 0

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
