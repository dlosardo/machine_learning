"""
Abstract class for cost function
"""

class CostFunction(object):
    """Cost function for machine learning algorithm
    will be implemented by a specific machine learning
    algorithm such as least squared error regression.
    """
    def __init__(self, hypothesis, targets):
        self.hypothesis = hypothesis
        self.targets = targets
