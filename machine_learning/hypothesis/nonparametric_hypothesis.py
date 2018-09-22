"""
Non Parametric Hypothesis.
hypothsis function is not a function of parameter
values to be estimated.
instead it performs a function on the data
"""
from machine_learning.hypothesis.hypothesis import Hypothesis


class NonParametricHypothesis(Hypothesis):
    def __init__(self, features, targets):
        """
        Constructor take both features and targets
        :param features: A nobs by 1 numpy array of feature variables
        :param: targets A nobs x ny numpy array of y values
        """
        super(NonParametricHypothesis, self).__init__(features)
        self.targets = targets
