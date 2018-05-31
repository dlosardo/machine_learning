import os
from flask import (
    Blueprint, render_template, redirect,
    url_for, session, request)  # , flash)
from werkzeug.utils import secure_filename
from webapp.extensions import data_uploads, db
from webapp.blueprints.user.models import User
from .forms import ModelForm
from .models import ModelRun
from machine_learning.driver.command_line_driver import run
from machine_learning.model_utils.factories import (
    HypothesisTypes, CostFunctionTypes, AlgorithmTypes,
    RegularizerNameFactory)

model_builder = Blueprint('model_builder', __name__,
                          template_folder='templates')


@model_builder.route('/models', methods=['GET', 'POST'])
def models():
    form = ModelForm()
    if form.validate_on_submit():
        # getting current user in session
        user = User.query.filter_by(username=session.get('name')).first()
        """
        looks like request.files['file'] (name of form)
        and file_form.file.data is the same...
        which to use?
        """
        # print ImmutableMultiDict object
        # print(request.files)
        # print FileStorage obj (next two are same)
        # print(request.files['data_form-file'])
        # print(form.data_form.file.data)
        # get FileStorage obj

        file = request.files['data_form-file']
        filename = secure_filename(file.filename)
        if ModelRun.duplicate_filename(user.id, filename):
            print("File already exists under this user, overwriting..")
            os.remove(data_uploads.path(filename))
        session['filename'] = filename
        f1 = data_uploads.save(file, name=filename)

        # obtaining all form info
        session['hypothesis'] = form.model_attribute_form.hypothesis.data
        session['regularizer'] = form.model_attribute_form.regularizer.data
        session['cost_function'] = form.model_attribute_form.cost_function.data
        session['algorithm'] = form.model_attribute_form.algorithm.data
        session['nfeatures'] = form.data_form.nfeatures.data
        session['ntargets'] = form.data_form.ntargets.data
        session['regularizer_weight'] = (form.hyperparam_form
                                         .regularizer_weight.data)
        session['tolerance'] = form.hyperparam_form.tolerance.data
        session['learning_rate'] = form.hyperparam_form.learning_rate.data

        # adding model run to database
        mr = ModelRun(hypothesis=form.model_attribute_form.hypothesis.data,
                      cost_function=(form.model_attribute_form
                                     .cost_function.data),
                      regularizer=form.model_attribute_form.regularizer.data,
                      algorithm=form.model_attribute_form.algorithm.data,
                      data_set_path=f1,
                      user_id=user.id,
                      data_url=data_uploads.url(f1))
        db.session.add(mr)
        db.session.commit()
        model_obj = run("webapp/static/data/" + session['filename'],
                        session['nfeatures'], session['ntargets'],
                        HypothesisTypes.get_type_from_number(int(session.get(
                            'hypothesis'))),
                        CostFunctionTypes.get_type_from_number(int(session.get(
                            'cost_function'))),
                        AlgorithmTypes.get_type_from_number(int(session.get(
                            'algorithm'))),
                        RegularizerNameFactory.get_regularizer_name(
                            int(session.get("regularizer"))),
                        session.get('regularizer_weight'),
                        session.get('learning_rate'),
                        session.get('tolerance'), None)
        # flash("Model successfully submitted and running...", "success")
        session['model_results'] = {'results':
                                    model_obj.get_results()}
        return redirect(url_for('.model_results'))

    return render_template('model_builder/models.html',
                           final=form,
                           hypothesis=session.get('hypothesis'),
                           regularizer=session.get('regularizer'),
                           cost_function=session.get('cost_function'),
                           algorithm=session.get('algorithm'),
                           filename=session.get('filename')
                           )


@model_builder.route('/results', methods=['GET', 'POST'])
def model_results():
    return render_template('model_builder/results.html',
                           m=session['model_results'])
