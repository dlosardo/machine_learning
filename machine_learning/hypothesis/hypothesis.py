"""
Hypothesis
The hypothesis is a function of the inputs (aka: x, features)
"""


class Hypothesis(object):
    def __init__(self, features):
        """
        Constructor for Hypothesis
        :param features: A nobs by 1 numpy array of feature variables
        """
        self.features = features
        self.nfeatures = self.features.shape[1]
        self.nobs = self.features.shape[0]

    def hypothesis_function(self, x):
        """
        Abstract method to compute the hypothesis function
        :returns: A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        raise NotImplementedError
