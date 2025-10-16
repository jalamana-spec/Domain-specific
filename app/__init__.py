import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

db = SQLAlchemy()

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    # ensure data & upload directories exist
    data_dir = app.config.get("DATA_DIR")
    if data_dir and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    uploads = os.path.join(data_dir, "uploads")
    if not os.path.exists(uploads):
        os.makedirs(uploads, exist_ok=True)

    db.init_app(app)

    with app.app_context():
        # import models so tables are registered
        from . import models
        # register routes
        from . import routes
        routes.init_app(app)
        # create tables if missing
        db.create_all()

    return app
