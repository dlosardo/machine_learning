from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired, Length


class UserForm(FlaskForm):
    name = StringField(
        'Name',
        validators=[DataRequired(), Length(max=255)]
    )
    password = StringField(
        'password',
        validators=[DataRequired(), Length(max=255, min=8)]
    )
    submit = SubmitField('Submit')
