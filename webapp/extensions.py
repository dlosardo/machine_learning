from flask_sqlalchemy import SQLAlchemy
from flask_uploads import UploadSet

db = SQLAlchemy()
data_uploads = UploadSet('data')
