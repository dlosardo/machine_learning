from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import (Form, SubmitField, RadioField,
                     IntegerField, FloatField, validators,
                     FormField)
from machine_learning.model_utils.factories import (
    HypothesisTypes, CostFunctionTypes, AlgorithmTypes, RegularizerTypes)
from machine_learning.model_utils.model_setup import ModelSetup
from machine_learning.utils.exceptions import (
    HypothesisCostFunctionDependencyException,
    CostFunctionAlgorithmDependencyException)


class DataForm(Form):
    file = FileField(validators=[FileRequired()])
    nfeatures = IntegerField("Number Features",
                             [validators.NumberRange(min=1)])
    ntargets = IntegerField("Number Targets",
                            [validators.NumberRange(min=1),
                             validators.optional()])


class ModelAttributeForm(Form):
    hypothesis = RadioField(
        'Hypothesis',
        choices=HypothesisTypes.tuple_pair()
    )
    cost_function = RadioField(
        'Cost Function',
        choices=CostFunctionTypes.tuple_pair()
    )
    algorithm = RadioField(
        'Algorithm',
        choices=AlgorithmTypes.tuple_pair()
    )
    regularizer = RadioField(
        'Regularizer',
        choices=RegularizerTypes.tuple_pair()
    )


class HyperParameterForm(Form):
    k = IntegerField(
        'k', [validators.NumberRange(min=1),
              validators.optional()])
    regularizer_weight = FloatField(
        'Regularizer Weight', [validators.NumberRange(min=0.0),
                               validators.optional()])
    learning_rate = FloatField(
        'Learning Rate', [validators.NumberRange(min=0.0),
                          validators.optional()])
    tolerance = FloatField(
        'Tolerance', [validators.NumberRange(min=0.0),
                      validators.optional()])


class ModelForm(FlaskForm):
    hyperparam_form = FormField(HyperParameterForm)
    model_attribute_form = FormField(ModelAttributeForm)
    data_form = FormField(DataForm)
    submit = SubmitField('Submit Model')

    def validate(self):
        if not super().validate():
            return False
        hypothesis_type = HypothesisTypes(
            int(self.model_attribute_form.hypothesis.data))
        cost_function_type = CostFunctionTypes(
            int(self.model_attribute_form.cost_function.data))
        algorithm_type = AlgorithmTypes(
            int(self.model_attribute_form.algorithm.data))
        try:
            model_setup_obj = ModelSetup(hypothesis_type,
                                         algorithm_type,
                                         cost_function_type
                                         )
            model_setup_obj.check_dependencies()
            return True
        except HypothesisCostFunctionDependencyException as e:
            self.model_attribute_form.hypothesis.errors = [e.message]
            return False
        except CostFunctionAlgorithmDependencyException as e:
            self.model_attribute_form.cost_function.errors = [e.message]
            return False


# TODO:
# What data should I store? It would be cool to show how the
#  cost function is being updated in real time. (websockets?)
# Would I use D3 for that?
# Final output would be the set of parameter estimates and
#  standard errors. Also model fit statistics.
# There should then be a way to allow users to upload data
#  to be predicted
# Use celery to manage tasks
