from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet
from flask_debugtoolbar import DebugToolbarExtension


db = SQLAlchemy()
data_uploads = UploadSet('data')
debug_toolbar = DebugToolbarExtension()
