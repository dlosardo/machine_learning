from werkzeug.security import generate_password_hash, check_password_hash
import datetime
from webapp.blueprints.model_builder.models import ModelRun
from webapp.extensions import db


class User(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255))
    password = db.Column(db.String(255))
    password_hash = db.Column(db.String(255))
    created = db.Column(db.DateTime(), default=datetime.datetime.now)
    model_runs = db.relationship(
        'ModelRun',
        backref='user',
        lazy='dynamic'
    )

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __repr__(self):
        return "<User '{}'>".format(self.username)

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
