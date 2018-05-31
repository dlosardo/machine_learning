from webapp.extensions import db


class ModelRun(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    user_id = db.Column(db.Integer(), db.ForeignKey('user.id'))
    hypothesis = db.Column(db.String(255))
    regularizer = db.Column(db.String(255))
    cost_function = db.Column(db.String(255))
    algorithm = db.Column(db.String(255))
    data_set_path = db.Column(db.String(255))
    data_url = db.Column(db.String(255))
    results = db.Column(db.Text())

    @classmethod
    def duplicate_filename(cls, user_id, data_set_path):
        if cls.query.filter_by(user_id=user_id,
                               data_set_path=data_set_path).first():
            return True
        else:
            return False
