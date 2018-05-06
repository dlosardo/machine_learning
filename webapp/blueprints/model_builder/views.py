from flask import (
    Blueprint, render_template, redirect,
    url_for, session, request)
# from werkzeug.utils import secure_filename
from webapp.extensions import data_uploads
from .forms import ModelForm, DataUploadForm

model_builder = Blueprint('model_builder', __name__,
                          template_folder='templates')


@model_builder.route('/models', methods=['GET', 'POST'])
def models():
    print(request)
    form = ModelForm()
    file_form = DataUploadForm()
    if file_form.validate_on_submit():
        print(request.files)
        """
        looks like request.files['file'] (name of form)
        and file_form.file.data is the same...
        which to use?
        """
        print(request.files['file'])
        f = file_form.file.data
        print(f)
        # filename = secure_filename(f.filename)
        session['f'] = f.mimetype
        f1 = data_uploads.save(request.files['file'])
        print(f1)
        return redirect(url_for('.models'))
    if form.validate_on_submit():
        session['hypothesis'] = form.hypothesis.data
        session['regularizer'] = form.regularizer.data
        session['cost_function'] = form.cost_function.data
        session['algorithm'] = form.algorithm.data
        return redirect(url_for('.models'))
    return render_template('model_builder/models.html', form=form,
                           file_form=file_form,
                           hypothesis=session.get('hypothesis'),
                           regularizer=session.get('regularizer'),
                           cost_function=session.get('cost_function'),
                           algorithm=session.get('algorithm'),
                           f=session.get('f')
                           )
    # here's where we would allow the user to select
    # various model features. Also need a file selector
    # to upload data.
    # Once a user selects these and uploads the data,
    # it has to be validated. If that passes, then the
    # model can run. The data can then be stored in a db.
    # maybe just the model parameters and variances?
