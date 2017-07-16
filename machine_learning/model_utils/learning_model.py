"""
Learning model consists of:
    Model side: Algorithm, which has a CostFunction, which has a Hypothesis
"""
MAX_ITER = 10000000

class LearningModel(object):
    def __init__(self, algorithm_obj):
        self.algorithm_obj = algorithm_obj
        self.niter = None
        self.loglikelihood = None
        self.fit_statistics = None

    def run_model(self):
        self.algorithm_obj.algorithm()
        if self.algorithm_obj.converged:
            parameter_variances_list = self.algorithm_obj.cost_function.parameter_variances()
            if parameter_variances_list is not None:
                self.algorithm_obj.cost_function.hypothesis.parameter_list.set_parameter_variances(parameter_variances_list)
                self.algorithm_obj.cost_function.hypothesis.parameter_list.set_covariance_matrix(self.algorithm_obj.cost_function.variance_covariance_matrix())
            self.algorithm_obj.cost_function.hypothesis.parameter_list.set_parameter_covariance_matrix_symbolic()

    def print_results(self):
        if self.algorithm_obj.converged:
            self.algorithm_obj.cost_function.hypothesis.parameter_list.print_parameters()
            print(self.algorithm_obj.cost_function.hypothesis.parameter_list.symbolic_covariance_matrix)

    def get_parameter_point_estimates(self):
        if self.algorithm_obj.converged:
            return self.algorithm_obj.cost_function.hypothesis.parameter_list.get_parameters()

    def predict(self):
        pass

    def plot(self):
        pass