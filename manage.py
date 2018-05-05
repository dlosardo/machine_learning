from flask_script import Manager, Server
from flask_migrate import Migrate, MigrateCommand
from webapp.app import create_app
from webapp.extensions import db
from webapp.blueprints.user.models import User
from webapp.blueprints.model_builder.models import ModelRun
from webapp.config import DevConfig

app = create_app(DevConfig)
migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command("server", Server())
manager.add_command("db", MigrateCommand)


@manager.shell
def make_shell_context():
    """
    Makes it possible to run the python shell
    python manage.py shell
    """
    return dict(app=app, db=db, User=User, ModelRun=ModelRun)


if __name__ == '__main__':
    """
    Call with:
    python manage.py server
    """
    manager.run()
