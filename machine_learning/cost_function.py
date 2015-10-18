"""
Abstract class for cost function
"""

class CostFunction(object):
    """Cost function for machine learning algorithm
    will be implemented by a specific machine learning
    algorithm such as least squared error regression.
    """
    def __init__(self, hypothesis, features, yvalues):
        self.hypothesis = hypothesis
        self.features = features
        self.yvalues = yvalues
