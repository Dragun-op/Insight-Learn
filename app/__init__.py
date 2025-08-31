from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_wtf import CSRFProtect
from dotenv import load_dotenv
import os
from .models import db, User

login_manager = LoginManager()
mail = Mail()
csrf = CSRFProtect()

def create_app():
    load_dotenv()

    app = Flask(__name__)
    app.config.from_object('config.Config')

    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "auth.login"

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from .auth.routes import auth as auth_bp
    from .nlp.routes import nlp as nlp_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(nlp_bp)

    return app