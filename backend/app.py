# app.py
from flask import Flask, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import os
from routes import register_blueprints
from extensions import mongo, login_manager

def create_app():
    load_dotenv()
    app = Flask(__name__)

    # --- core config ---
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")
    app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/auth_app")
    app.config["CORS_SUPPORTS_CREDENTIALS"] = True
    app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME", "your_session_cookie_name")
    app.config["UPLOAD_FOLDER"] = os.getenv("UPLOAD_FOLDER", "uploads")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # --- init extensions ---
    mongo.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # return 401 JSON instead of redirecting (helps with CORS + SPAs)
    @login_manager.unauthorized_handler
    def _unauth():
        return jsonify({"error": "Unauthorized"}), 401

    # --- CORS ---
    allowed = [
        os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        "http://127.0.0.1:3000",
        "http://localhost:3001",           # keep if you sometimes run on 3001
        "http://127.0.0.1:3001",
    ]
    CORS(
        app,
        resources={r"/api/*": {"origins": allowed}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    )

    # --- blueprints ---
    register_blueprints(app)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)),
            debug=bool(int(os.getenv("FLASK_DEBUG", "1"))))