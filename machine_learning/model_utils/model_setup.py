"""
Setup for model components.
Hypothesis, CostFunction, Algorithm
"""
from machine_learning.model_utils.factories import HypothesisFactory, CostFunctionFactory, AlgorithmFactory

class ModelSetup(object):
    def __init__(self, hypothesis_name, cost_function_name, algorithm_name):
        self.hypothesis_name = hypothesis_name
        self.cost_function_name = cost_function_name
        self.algorithm_name = algorithm_name
        self.check_dependencies()

    def check_dependencies(self):
        if self.hypothesis_name == "multiple_linear_regression":
            if self.cost_function_name == "log_loss":
                raise Exception("Cannot use {} cost function with {} hypothesis".format(
                    self.cost_function_name, self.hypothesis_name))

    def model_setup(self, features, targets=None, learning_rate=None, tolerance=None
            , starting_parameter_values=None):
        hypothesis_obj = self.set_hypothesis(features)
        optional_cost_fnx_arguments = [{"targets": targets}]
        cost_fnx_kwargs = {}
        for optional_cost_fnx_argument in optional_cost_fnx_arguments:
            for key, val in optional_cost_fnx_argument.items():
                if val is not None:
                    cost_fnx_kwargs.update(optional_cost_fnx_argument)
        cost_function_obj = self.set_cost_function(hypothesis_obj, **cost_fnx_kwargs)
        optional_arguments = [{"learning_rate": learning_rate}, {"tolerance": tolerance}
                , {"starting_parameter_values": starting_parameter_values}]
        algorithm_kwargs = {key:val
                            for optional_argument in optional_arguments
                            for key, val in optional_argument.items()
                            if val is not None}
        algorithm_obj = self.set_algorithm(cost_function_obj, **algorithm_kwargs)
        return algorithm_obj

    def set_hypothesis(self, features, **kwargs):
        hypothesis_object = HypothesisFactory.get_hypothesis_by_name(self.hypothesis_name, features, **kwargs)
        return hypothesis_object

    def set_cost_function(self, hypothesis, targets=None, **kwargs):
        cost_function_object = CostFunctionFactory.get_cost_function_by_name(
                self.cost_function_name, hypothesis, targets, **kwargs)
        return cost_function_object

    def set_algorithm(self, cost_function, **kwargs):
        algorithm_object = AlgorithmFactory.get_algorithm_by_name(self.algorithm_name
                , cost_function=cost_function
                , **kwargs)
        return algorithm_object
