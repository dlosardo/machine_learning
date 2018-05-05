from flask_wtf import FlaskForm
from wtforms import SubmitField, RadioField
from machine_learning.model_utils.factories import (
    HypothesisTypes, CostFunctionTypes, AlgorithmTypes, RegularizerTypes)


class ModelForm(FlaskForm):
    hypothesis = RadioField(
        'Hypothesis',
        choices=list(zip(map(lambda x: str(x),
                             HypothesisTypes.values_list()),
                         map(lambda x: x.replace("_", " "),
                             HypothesisTypes.names_list())))
    )
    cost_function = RadioField(
        'Cost Function',
        choices=list(zip(map(lambda x: str(x),
                             CostFunctionTypes.values_list()),
                         map(lambda x: x.replace("_", " "),
                             CostFunctionTypes.names_list())))
    )
    algorithm = RadioField(
        'Algorithm',
        choices=list(zip(map(lambda x: str(x),
                             AlgorithmTypes.values_list()),
                         map(lambda x: x.replace("_", " "),
                             AlgorithmTypes.names_list())))
    )
    regularizer = RadioField(
        'Regularizer',
        choices=list(zip(map(lambda x: str(x),
                             RegularizerTypes.values_list()),
                         map(lambda x: x.replace("_", " "),
                             RegularizerTypes.names_list())))
    )
    submit = SubmitField('Submit Model')

# TODO: Make a form for inputting data
# Once a user has chosen model attributes and uploaded data,
#  the model run can begin. There probably needs to be
#  another form that says "Run Model", and that triggers
#  the actual model running.
# What data should I store? It would be cool to show how the
#  cost function is being updated in real time.
# Would I use D3 for that? don't think matplotlib would work
# If D3, I would have to pass in the values somehow...
# maybe through an API?
# Final output would be the set of parameter estimates and
#  standard errors. Also model fit statistics.
# There should then be a way to allow users to upload data
#  to be predicted.


class RunModelForm(FlaskForm):
    run_model = SubmitField('Run Model')
