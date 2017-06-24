"""
Abstract class for cost function
This is a function of the parameters.
The list of parameters are found in the Hypothesis object
"""

class CostFunction(object):
    """Cost function for machine learning algorithm
    will be implemented by a specific machine learning
    algorithm such as least squared error regression.
    """
    def __init__(self, hypothesis, targets):
        self.hypothesis = hypothesis
        self.targets = targets
        self.nobs = self.targets.shape[0]

    def initialize_parameters(self, param_dict=None):
        """Initializes the parameter values
        :param param_dict A dictionary with the form parameter_name: parameter_value
        """
        self.hypothesis.initialize_parameters(param_dict)

    def update_parameters(self, param_array):
        """Updates parameter values
        :param param_array A numpy array of dimension nparams x 1 consisting of
         parameter values.
        """
        self.hypothesis.update_parameters(param_array)

    def get_parameters(self):
        """Gets parameter values
        :return A nparam x 1 numpy array of parameter values
        """
        return self.hypothesis.get_parameters()

    def cost_function(self):
        raise NotImplementedError
