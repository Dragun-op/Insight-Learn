from flask import Blueprint

def create_auth_blueprint():
    from .routes import auth
    return auth