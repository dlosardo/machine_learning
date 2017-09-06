"""
Abstract class for cost function
Has a hypothesis and targets
This is a function of the parameters.
The list of parameters are found in the Hypothesis object
"""


class CostFunction(object):
    def __init__(self, hypothesis, targets=None):
        """
        Cost function for machine learning algorithm
        will be implemented by a specific machine learning
        algorithm such as least squared error regression.
        """
        self.hypothesis = hypothesis
        self.targets = targets
        self.nobs = self.targets.shape[0]

    def initialize_parameters(self, param_dict=None):
        """
        Initializes the parameter values
        :param param_dict: A dictionary with the form parameter_name: parameter_value
        """
        self.hypothesis.initialize_parameters(param_dict)

    def update_parameters(self, param_array):
        """
        Updates parameter values
        :param param_array: A numpy array of dimension nparams x 1 consisting of
         parameter values.
        """
        self.hypothesis.update_parameters(param_array)

    def get_parameters(self):
        """
        Gets parameter values
        :returns: A nparam x 1 numpy array of parameter values
        """
        return self.hypothesis.get_parameters()

    def reset(self):
        """
        Resets the cost function and hypothesis
        """
        self.hypothesis.reset()

    def cost_function(self):
        """
        Computes the cost function
        :returns: a 1 x 1 numpy array containing a float value representing the
         value of the cost function
        """
        raise NotImplementedError

    def cost_function_derivative(self):
        """
        This calcualtes the partial derivative of the cost function
        with respect to the parameters
        :returns: A nparam x 1 np array of the values of the derivatives of the parameters
        """
        raise NotImplementedError

    def convergence_criteria_met(self, current_cost, new_cost, tolerance):
        """
        Calculates whether the convergence criteria has been met.
        In general is calculated as: |f(x_i) - f(x_i -1)| < tolerance
        where f(x_i) is the cost at iteration i and f(x_i -1) is the cost at iteration i-1.
        :param current_cost: A 1x1 numpy array with the first element a float representing the current cost associated with the algorithm
        :param new_cost: A 1x1 numpy array with the first element a float representing the new cost associated with the algorithm
        :param tolerance: A float value specifying the point at which convergence has been reached
        :returns: True if the convergence criteria has been met, false otherwise
        """
        raise NotImplementedError

    def convergence_value(self, current_cost, new_cost):
        """
        Calculates the convergence value
        :param current_cost: A 1x1 numpy array with the first element a float representing the current cost associated with the algorithm
        :param new_cost: A 1x1 numpy array with the first element a float representing the new cost associated with the algorithm
        :returns: A float value representing the convergence value
        """
        raise NotImplementedError

    def variance_covariance_matrix(self):
        return None

    def standard_errors(self):
        return None

    def parameter_variances(self):
        return None
