"""
Logistic Regression
IS a Hypothesis
Matrix form:
    hypothesis: 1. / 1. + exp(-1. * X%*%THETA)
    where %*% is matrix multiplication, X is a matrix of dimension
    nobs x nparams (nx + 1), and THETA is a matrix
        of dimension nparms (nx + 1) x 1
"""
from numpy import exp
from machine_learning.hypothesis.regression import Regression


class LogisticRegression(Regression):
    """
    Constructor creates an intercept Parameter object and
    the applicable number of slope Parameter objects
    """
    def __init__(self, features):
        super(LogisticRegression, self).__init__(features)
        """
        We know we have an intercept but not sure how many slope
            parameters need to be created yet
        :param features A nobs x nx numpy array of feature values
        """
        self.set_parameters()

    def hypothesis_function(self, data=None):
        """
        Computes the hypothesis function.
        Estimated probability that y=1 on input x
        P(y=1|x;theta) = 1/(1+exp(-theta*features))
        features are a vector of feature values of dimension nobs x 1
        :returns: A matrix of dimension nobs x 1 with the results of the
         hypothesis computation.
        """
        if data is not None:
            return 1. / (1. + exp(-1. * data.dot(self.get_parameters())))
        else:
            return 1. / (1. + exp(-1. * self.features.dot(
                self.get_parameters())))

    def decision_boundary(self):
        """
        Computes the equation for a decision boundary
        theta*features >= 0 when y = 1
        theta*features < 0 when y = 0
        theta[1:nparams]*features_no_intercept >= theta[0]*1

        solve for theta*features = 0
        """
        # TODO: implement and plot
        pass
