"""
Setup for model components.
Hypothesis, CostFunction, Algorithm
"""
from machine_learning.model_utils.factories import (
    HypothesisFactory, CostFunctionFactory, AlgorithmFactory,
    HypothesisCostFunctions, CostFunctionsAlgorithms)
from machine_learning.utils.exceptions import (
    HypothesisCostFunctionDependencyException,
    CostFunctionAlgorithmDependencyException)


class ModelSetup(object):
    def __init__(self, hypothesis_type, cost_function_type, algorithm_type):
        """
        Given a hypothesis type, cost function type, and algorithm type:
            1. Checks whether the dependencies between the
                different model component are valid
            2. Sets each model component
        """
        self.hypothesis_type = hypothesis_type
        self.cost_function_type = cost_function_type
        self.algorithm_type = algorithm_type
        self.check_dependencies()

    def check_dependencies(self):
        """
        Where the model component dependencies are validated.
        """
        if self.cost_function_type not in (
           HypothesisCostFunctions.cost_function_hypothesis_dict()[
               self.hypothesis_type]):
            raise HypothesisCostFunctionDependencyException(
                self.hypothesis_type.name, self.cost_function_type.name)
        if self.algorithm_type not in (
           CostFunctionsAlgorithms.cost_function_algorithm_dict()[
               self.cost_function_type]):
            raise CostFunctionAlgorithmDependencyException(
                self.cost_function_type.name, self.algorithm_type.name)

    def model_setup(self, features, targets=None, regularizer_name=None,
                    regularization_weight=None, learning_rate=None,
                    tolerance=None, starting_parameter_values=None):
        """
        Performs the model setup. First sets hypothesis, then cost function,
            then algorithm.
        :returns: An algorithm object
        """
        hypothesis_obj = self.set_hypothesis(features)
        optional_cost_fnx_arguments = [
            {"targets": targets},
            {"regularizer_name": regularizer_name},
            {"regularization_weight": regularization_weight}]
        cost_fnx_kwargs = {}
        for optional_cost_fnx_argument in optional_cost_fnx_arguments:
            for key, val in optional_cost_fnx_argument.items():
                if val is not None:
                    cost_fnx_kwargs.update(optional_cost_fnx_argument)
        cost_function_obj = self.set_cost_function(
            hypothesis_obj, **cost_fnx_kwargs)
        optional_arguments = [
            {"learning_rate": learning_rate}, {"tolerance": tolerance},
            {"starting_parameter_values": starting_parameter_values}]
        algorithm_kwargs = {key: val
                            for optional_argument in optional_arguments
                            for key, val in optional_argument.items()
                            if val is not None}
        algorithm_obj = self.set_algorithm(
            cost_function_obj, **algorithm_kwargs)
        return algorithm_obj

    def set_hypothesis(self, features, **kwargs):
        hypothesis_object = HypothesisFactory.get_hypothesis(
            self.hypothesis_type.value, features, **kwargs)
        return hypothesis_object

    def set_cost_function(self, hypothesis, targets=None, **kwargs):
        cost_function_object = CostFunctionFactory.get_cost_function(
                self.cost_function_type.value, hypothesis, targets, **kwargs)
        return cost_function_object

    def set_algorithm(self, cost_function, **kwargs):
        algorithm_object = AlgorithmFactory.get_algorithm(
            self.algorithm_type.value, cost_function=cost_function, **kwargs)
        return algorithm_object
