"""
Abstract class for iterative parametric algorithm
"""
from machine_learning.algorithm.parametric_algorithm import ParametricAlgorithm


class IterativeParametricAlgorithm(ParametricAlgorithm):
    def __init__(self, cost_function, param_starting_values):
        """
        Iterative Parametric Algorithm.
        :param cost_function: A CostFunction object, e.g., SquaredErrorLoss
        :param param_starting_values: A dict of the form:
            {parameter_name: parameter_value}
        """
        super(IterativeParametricAlgorithm, self).__init__(
            cost_function, param_starting_values)
        self.iter = 0
        self.converged = False
        self.cost_function_list = []
        self.convergence_value_list = []

    def algorithm(self):
        """
        Run the algorithm
        Call iterate until the the convergence criteria is met
        """
        while (not self.converged):
            self.iter = self.iter + 1
            self.iterate()
            self.cost_function_list.append(self.cost_function.cost_function())
            self.convergence_value_list.append(self.convergence_value())
            if self.iter % 100000 == 0:
                print("iter: {}".format(self.iter))
                print("current convergence value: {}".format(
                    self.convergence_value()))
                print("Parameters: ")
                print(self.get_parameters())
                print("cost function : {}".format(
                    self.cost_function.cost_function()))
            if self.convergence_criteria_met():
                self.converged = True

    def reset(self):
        raise NotImplementedError

    def iterate(self):
        raise NotImplementedError

    def convergence_criteria_met(self):
        raise NotImplementedError

    def convergence_value(self):
        raise NotImplementedError
