"""
Abstract class for supervised algorithm
"""
from machine_learning.algorithm.supervised_algorithm import SupervisedAlgorithm

class IterativeSupervisedAlgorithm(SupervisedAlgorithm):
    """Supervised algorithm.
    :param cost_function A CostFunction object, e.g., SquaredErrorLoss
    :param param_starting_values A dict of the form parameter_name: parameter_value
    """
    def __init__(self, cost_function, param_starting_values):
        super(IterativeSupervisedAlgorithm, self).__init__(cost_function, param_starting_values)
        self.iter = 0
        self.converged = False
        self.initialize_parameters()

    def algorithm(self):
        """Run the algorithm
        Call iterate until the change in the cost function is less than the specified tolerance value
        """
        while (True):
            self.iterate()
            if self.iter % 100000 == 0:
                print("iter: {}".format(self.iter))
                print("current convergence value: {}".format(self.convergence_value()))
                print("Parameters: ")
                print(self.get_parameters())
                print("cost function : {}".format(self.cost_function.cost_function()))
            if self.convergence_criteria_met():
                self.converged = True
                break
            self.iter = self.iter + 1

    def reset(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def convergence_criteria_met(self):
        raise NotImplementedError

    def convergence_value(self):
        raise NotImplementedError
