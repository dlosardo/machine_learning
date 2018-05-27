from flask import Blueprint, render_template, redirect, url_for, session
from webapp.extensions import db
from .models import User
from .forms import UserForm, SimpleSignInForm

user = Blueprint('user', __name__, template_folder='templates')


@user.route('/user_list')
def all_users(page=1):
    users = User.query.order_by(User.created.desc()
                                ).paginate(page, 10)
    return render_template('user/user_list.html', users=users)


@user.route('/users', methods=['GET', 'POST'])
def users():
    form = UserForm()
    if form.validate_on_submit():
        user_form = form.name.data
        user = User.query.filter_by(username=form.name.data).first()
        if user is None:
            user = User(username=user_form, password=form.password.data)
            db.session.add(user)
            db.session.commit()
        session['name'] = user_form
        form.name.data = ''
        form.password.data = ''
        return redirect(url_for('.users'))
    return render_template('user/users.html', form=form, name=session.get(
        'name'), password=session.get('password'))


@user.route('/user/<username>')
def show_user_info(username):
    user = User.query.filter_by(username=username).first()
    if user:
        return "User {} was created on {}".format(user.username, user.created)
    else:
        return "User {} not found".format(username)


@user.route('/simple_sign_in', methods=['GET', 'POST'])
def simple_sign_in():
    signin_form = SimpleSignInForm()
    if signin_form.validate_on_submit():
        session['name'] = signin_form.name.data
        return redirect(url_for('.simple_sign_in'))
    return render_template('user/simple_sign_in.html',
                           form=signin_form, name=session.get('name'))
