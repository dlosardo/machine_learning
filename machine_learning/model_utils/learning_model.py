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
        self.converged = False

    def run_model(self):
        self.algorithm_obj.algorithm()
        if self.algorithm_obj.converged:
            self.converged = True
            self.niter = self.algorithm_obj.iter
            parameter_variances_array = (
                self.algorithm_obj.cost_function.parameter_variances())
            if parameter_variances_array is not None:
                (self.algorithm_obj.cost_function.hypothesis.parameter_list.
                 set_parameter_variances(
                    list(parameter_variances_array)))
                (self.algorithm_obj.cost_function.hypothesis.parameter_list.
                 set_covariance_matrix(
                    (self.algorithm_obj.cost_function.
                     variance_covariance_matrix())))
            (self.algorithm_obj.cost_function.hypothesis.parameter_list.
             set_parameter_covariance_matrix_symbolic())

    def print_results(self):
        if self.converged:
            (self.algorithm_obj.cost_function.hypothesis.parameter_list.
             print_parameters())
            (print(self.algorithm_obj.cost_function.hypothesis.parameter_list.
                   symbolic_covariance_matrix))

    def get_parameter_point_estimates(self):
        if self.converged:
            return (
                self.algorithm_obj.cost_function.
                hypothesis.parameter_list.get_parameters())

    def predict(self, new_values):
        if self.converged:
            if new_values is None:
                pass
            pass

    def plot(self):
        pass

    def plot_convergence_values():
        pass

    def plot_cost_function_values():
        pass
