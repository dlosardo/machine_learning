from flask import Blueprint, render_template, redirect, url_for, session
from .forms import ModelForm

model_builder = Blueprint('model_builder', __name__,
                          template_folder='templates')


@model_builder.route('/models', methods=['GET', 'POST'])
def models():
    form = ModelForm()
    if form.validate_on_submit():
        session['hypothesis'] = form.hypothesis.data
        session['regularizer'] = form.regularizer.data
        session['cost_function'] = form.cost_function.data
        session['algorithm'] = form.algorithm.data
        print(form.hypothesis)
        return redirect(url_for('.models'))
    return render_template('model_builder/models.html', form=form,
                           hypothesis=session.get('hypothesis'),
                           regularizer=session.get('regularizer'),
                           cost_function=session.get('cost_function'),
                           algorithm=session.get('algorithm')
                           )
    # here's where we would allow the user to select
    # various model features. Also need a file selector
    # to upload data.
    # Once a user selects these and uploads the data,
    # it has to be validated. If that passes, then the
    # model can run. The data can then be stored in a db.
    # maybe just the model parameters and variances?
