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
    # Get frontend origin from environment, with fallbacks
    frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:80")
    allowed = [
        frontend_origin,
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",           # keep if you sometimes run on 3001
        "http://127.0.0.1:3001",
        "http://localhost:80",
        "http://127.0.0.1:80",
        "http://localhost",                # Allow localhost without port
        "http://127.0.0.1",                # Allow 127.0.0.1 without port
    ]
    CORS(
        app,
        resources={r"/api/*": {"origins": allowed}},
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        expose_headers=["Content-Type"],
        max_age=3600,
    )

    # --- blueprints ---
    register_blueprints(app)
    
    # --- health check endpoint ---
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint for Docker healthchecks"""
        try:
            # Check MongoDB connection
            mongo.db.command('ping')
            return jsonify({"status": "healthy", "mongodb": "connected"}), 200
        except Exception as e:
            return jsonify({"status": "unhealthy", "mongodb": "disconnected", "error": str(e)}), 503
    
    # --- seed public datasets on startup ---
    # Seed datasets after MongoDB is initialized
    # Note: This is deferred to avoid blocking app startup
    # The seeding will happen in the background and won't fail app startup
    def _seed_datasets_safe():
        import time
        # Wait a bit for MongoDB to be ready
        time.sleep(2)
        try:
            with app.app_context():
                # Check if MongoDB is available before seeding
                try:
                    mongo.db.command('ping')
                except Exception as e:
                    print(f"MongoDB not ready, skipping dataset seeding: {e}")
                    return
                
                from seed_datasets import seed_public_datasets
                seed_public_datasets(app.config["UPLOAD_FOLDER"])
        except Exception as e:
            # Don't fail app startup if seeding fails
            print(f"Warning: Could not seed public datasets: {e}")
            import traceback
            traceback.print_exc()
    
    # Run seeding in background thread to avoid blocking startup
    import threading
    seeding_thread = threading.Thread(target=_seed_datasets_safe, daemon=True)
    seeding_thread.start()
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)),
            debug=bool(int(os.getenv("FLASK_DEBUG", "1"))))