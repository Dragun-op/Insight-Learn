from flask import Blueprint

def create_nlp_blueprint():
    from .routes import nlp
    return nlp