"""
Perceptron is a Hypothesis
    z = features_i%*%WEIGHTS
    y = 1 if z>0 else 0
"""
from numpy import append, ones
from machine_learning.hypothesis.hypothesis import Hypothesis
from machine_learning.model_utils.parameter import Parameter, ParameterList
from machine_learning.utils.exceptions import IncorrectMatrixDimensions


class Perceptron(Hypothesis):
    """
    """
    def __init__(self, features):
        super(Perceptron, self).__init__(features)
        self.nparams = features.shape[1] + 1
        self.bias = Parameter(name="bias", value=None, default_starting_value=0.)
        self.parameter_list.add_parameter(self.bias)
        for i in range(0, features.shape[1]):
            tmp_feature = Parameter(name="feature_{}".format(i), value=None, default_starting_value=0.)
            self.parameter_list.add_parameter(tmp_feature)
        if self.features.shape[1] != self.nparams - 1:
            raise IncorrectMatrixDimensions(
                "Number of columns is equal to %d but should be equal to %d" % self.features.shape[1], self.nparams - 1)
        # Next line adds a vector of 1s indicating the bias.
        # self.features becomes a matrix of dimension nobs x nx with the first column
         # consisting of 1s and the next columns consisting of x values.
        self.features = append(ones(self.features.shape[0]).reshape(self.features.shape[0], 1), self.features, 1)

    def hypothesis_function(self, index):
        """
        Computes the hypothesis function.
        current feature vector * parameters
        features are a vector of feature values of dimension nobs x 1
        :returns A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        z = self.features[index, :].dot(self.get_parameters())
        y = z
        y[y > 0] = 1
        y[y <= 0] = 0
        return y
