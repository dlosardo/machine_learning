"""
Abstract class for machine learning algorithm
"""


class MachineLearningAlgorithm(object):
    def __init__(self):
        pass

    def algorithm(self):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def print_results(self):
        raise NotImplementedError
