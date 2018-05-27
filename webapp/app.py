import os
from flask import Flask
from flask_uploads import configure_uploads

from webapp.extensions import db, data_uploads
from webapp.blueprints.main import main
from webapp.blueprints.user import user
from webapp.blueprints.model_builder import model_builder

config = {
    "development": "webapp.config.DevConfig",
    "testing": "webapp.config.TestingConfig",
    "default": "webapp.config.DevConfig"
}


def create_app(settings_override=None):
    """
    Create a Flask application using the app factory pattern.
    :param settings_override Passedd if default settings should be overidden
    :return Flask application instance
    """
    app = Flask(__name__, instance_relative_config=True)
    configure_app(app, settings_override)
    app.register_blueprint(main)
    app.register_blueprint(user)
    app.register_blueprint(model_builder)
    extensions(app)
    return app


def configure_app(app, settings_override):
    config_name = os.getenv('FLASK_CONFIGURATION', 'default')
    app.config.from_object(config[config_name])
    app.config.from_pyfile(
        'config.cfg', silent=True)  # instance-folders configuration
    # print(app.config)
    if settings_override:
        app.config.from_object(config[settings_override])
    return None


def extensions(app):
    """
    Register extensions (mutates the app passed in)
    :param app: Flask application instance
    :return None
    """
    db.init_app(app)
    configure_uploads(app, data_uploads)
    return None
