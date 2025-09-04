from .auth import auth_bp
from .classification import classification_bp
from .classify import classify_bp
from .datasets import datasets_bp
from .explanations import explanations_bp
from .predictions import predictions_bp
from .ratings import ratings_bp
from .trustworthiness import trust_bp
from .classify_and_explain import classify_and_explain_bp


def register_blueprints(app):
    app.register_blueprint(auth_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(ratings_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(classification_bp)
    app.register_blueprint(trust_bp)
    app.register_blueprint(classify_bp)
    app.register_blueprint(explanations_bp)
    app.register_blueprint(classify_and_explain_bp)