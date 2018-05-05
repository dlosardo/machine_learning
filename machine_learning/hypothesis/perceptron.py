"""
Perceptron is a Hypothesis
    z = features_i%*%WEIGHTS
    y = 1 if z>0 else 0
"""
from numpy.linalg import norm
from machine_learning.hypothesis.hypothesis import Hypothesis
from machine_learning.model_utils.parameter import Parameter
from machine_learning.utils.math_utils import add_constant


class Perceptron(Hypothesis):
    """
    """
    def __init__(self, features):
        super(Perceptron, self).__init__(features)
        self.set_parameters()
        # Next line adds a vector of 1s indicating the bias.
        # self.features becomes a matrix of dimension nobs x nx
        # with the first column
        # consisting of 1s and the next columns consisting of x values.
        self.features = add_constant(self.features)

    def set_parameters(self):
        self.nparams = self.nfeatures + 1
        bias = Parameter(
            name="bias", value=None, variance=None, default_starting_value=0.)
        self.parameter_list.add_parameter(bias)
        for i in range(0, self.features.shape[1]):
            tmp_weight = Parameter(
                name="weight_{}".format(i), value=None,
                variance=None, default_starting_value=0.)
            self.parameter_list.add_parameter(tmp_weight)

    def hypothesis_function(self, index):
        """
        Computes the hypothesis function.
        current feature vector * parameters
        features are a vector of feature values of dimension nobs x 1
        :returns: A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        z = self.features[index, :].dot(self.get_parameters())
        # activation
        y = z
        y[z > 0] = 1
        y[z <= 0] = 0
        return y

    def geometric_margin(self, index):
        """
        Calculates the distance of the input from the hyperplane
        gm_i = weights%*%x_i / ||weights||
        """
        gm = self.features[index, :].dot(
            self.get_parameters()) / norm(self.get_parameters())
        return gm
