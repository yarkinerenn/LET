# app.py
from flask import Flask, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from flask_apscheduler import APScheduler
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
    
    # --- init scheduler ---
    scheduler = APScheduler()
    scheduler.init_app(app)

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
    
    # --- seed public datasets on startup ---
    # Seed datasets after MongoDB is initialized
    with app.app_context():
        try:
            from seed_datasets import seed_public_datasets
            seed_public_datasets(app.config["UPLOAD_FOLDER"])
        except Exception as e:
            print(f"Warning: Could not seed public datasets: {e}")
        
        # --- setup scheduled model updates ---
        try:
            from services.model_updater import update_all_models
            
            # Schedule daily model updates at 2 AM
            def scheduled_model_update():
                with app.app_context():
                    try:
                        update_all_models(use_api_keys=True)
                        print("Scheduled model update completed successfully")
                    except Exception as e:
                        print(f"Error in scheduled model update: {e}")
            
            scheduler.add_job(
                id='update_models',
                func=scheduled_model_update,
                trigger='cron',
                hour=2,
                minute=0
            )
            
            # Run initial model update on startup (optional - can be disabled)
            try:
                update_all_models(use_api_keys=True)
                print("Initial model cache update completed")
            except Exception as e:
                print(f"Warning: Could not perform initial model update: {e}")
        except Exception as e:
            print(f"Warning: Could not setup model updater scheduler: {e}")
    
    # Start scheduler after all jobs are added
    scheduler.start()
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)),
            debug=bool(int(os.getenv("FLASK_DEBUG", "1"))))