from flask import Flask, jsonify, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from transformers import pipeline
from dotenv import load_dotenv
import openai
import os
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")  # Store API key in .env

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/auth_app'
app.config['CORS_SUPPORTS_CREDENTIALS'] = True

CORS(app, origins=["http://localhost:3000"], supports_credentials=True)
mongo = PyMongo(app)
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']

    @staticmethod
    def get(user_id):
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user_data:
            return None
        return User(user_data)

    @staticmethod
    def find_by_email(email):
        user_data = mongo.db.users.find_one({'email': email})
        if not user_data:
            return None
        return User(user_data)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if mongo.db.users.find_one({'email': data['email']}):
        return jsonify({"error": "Email already exists"}), 400

    password_hash = generate_password_hash(data['password'])
    user_data = {
        'username': data['username'],
        'email': data['email'],
        'password_hash': password_hash
    }
    result = mongo.db.users.insert_one(user_data)
    return jsonify({
        "message": "User created successfully",
        "id": str(result.inserted_id)
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user_data = mongo.db.users.find_one({'email': data['email']})
    if not user_data or not check_password_hash(user_data['password_hash'], data['password']):
        return jsonify({"error": "Invalid email or password"}), 401

    user = User(user_data)
    login_user(user)
    return jsonify({
        "message": "Logged in successfully",
        "user": {
            "id": str(user_data['_id']),
            "username": user_data['username']
        }
    })

@app.route('/api/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "user": {
                "id": current_user.id,
                "username": current_user.username
            }
        })
    return jsonify({"authenticated": False})
# Load DistilBERT model
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.json
        text = data.get('text', '')
        print(text)

        if len(text) < 3:
            return jsonify({"error": "Text must be at least 3 characters"}), 400

        # Classify text
        result = classifier(text)[0]

        # Store in MongoDB
        prediction = mongo.db.predictions.insert_one({
            "text": text,
            "label": result['label'],
            "score": result['score'],
            "timestamp": datetime.now()
        })

        return jsonify({
            "id": str(prediction.inserted_id),
            "label": result['label'],
            "score": result['score']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def generate_explanation():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')

        if not prediction_id:
            return jsonify({"error": "Missing prediction_id"}), 400

        prediction = mongo.db.predictions.find_one(
            {"_id": ObjectId(prediction_id)}
        )

        if not prediction:
            print('cant find')
            return jsonify({"error": "Prediction not found"}), 404

        prompt = f"""
        Explain this sentiment analysis result in simple terms:
        
        Text: {prediction['text']}
        Sentiment: {prediction['label']} ({prediction['score']*100:.1f}% confidence)
        
        Focus on key words and overall tone.
        Keep explanation under 3 sentences.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response["choices"][0]["message"]["content"] if "choices" in response else "No explanation generated."

        return jsonify({"explanation": explanation})

    except Exception as e:
        print(f"Error: {e}")  # Debugging
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(port=5000, debug=True)