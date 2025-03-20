from flask import Flask, jsonify, request
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from transformers import pipeline,AutoModelForSequenceClassification,AutoTokenizer
from dotenv import load_dotenv
from openai import OpenAI
from collections import Counter
import datasets
import os
import numpy as np
import shap
import io

import torch
load_dotenv()
# Store API key in .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
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
def explain_prediction():
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        text = data.get('text')
        explainer_type = data.get('explainer_type', 'llm')

        if not prediction_id or not text:
            return jsonify({"error": "Missing prediction_id or text"}), 400

        prediction = mongo.db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        if explainer_type == 'shap':
            explanation_data = generate_shap_explanation(text, prediction['label'])
            return jsonify({'explanation': explanation_data,'explainer_type': explainer_type})
        else:
            explanation_text = generate_llm_explanation(text, prediction['label'], prediction['score'])
            return jsonify({"explanation": explanation_text,'explainer_type': explainer_type})

    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_llm_explanation(text, label,score):
    try:



        prompt = f"""
        Explain this sentiment analysis result in simple terms:
        
        Text: {text}
        Sentiment: {label} ({score}% confidence)
        
        Focus on key words and overall tone.
        Keep explanation under 3 sentences.
        """

        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}])

        explanation = response.choices[0].message.content
        print(response)

        return explanation

    except Exception as e:
        print(f"Error: {e}")  # Debugging
        return jsonify({"error": str(e)}), 500

def generate_shap_explanation(input_text, label):
    """Generate an explanation using SHAP values"""
    try:
        pmodel = shap.models.TransformersPipeline(classifier, rescale_to_logits=False)
        pmodel([input_text])
        explainer2 = shap.Explainer(pmodel)
        shap_values2 = explainer2([input_text])

        output_str = get_top_phrases(shap_values2, top_n=10)

        plot=shap.plots.text(shap_values2[:, :, 1],display=False)

        return plot
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error in SHAP explanation: {str(e)}")
        return f"Could not generate SHAP explanation: {str(e)}"

@app.route('/api/classifications', methods=['GET'])
def get_classifications():
    try:
        # Retrieve last 10 classifications (modify limit as needed)
        predictions = mongo.db.predictions.find().sort("timestamp", -1).limit(10)

        # Convert ObjectId to string and prepare JSON response
        results = []
        for prediction in predictions:
            results.append({
                "id": str(prediction["_id"]),
                "text": prediction["text"],
                "label": prediction["label"],
                "score": round(prediction["score"] * 100, 1),  # Convert score to percentage
                "timestamp": prediction["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify({"classifications": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
def get_top_phrases(shap_values_obj, instance_idx=0, class_idx=1, top_n=5):
    # Extract text segments and SHAP values
    text_segments = shap_values_obj[instance_idx].data
    shap_vals = shap_values_obj[instance_idx].values[:, class_idx]

    # Pair and sort by absolute impact
    paired = sorted(zip(text_segments, shap_vals),
                    key=lambda x: -abs(x[1]))

    # Separate positive and negative impacts
    positive = [(text, val) for text, val in paired if val > 0]
    negative = [(text, val) for text, val in paired if val < 0]

    # Build result string
    result = [
        f"Top {top_n} impactful phrases for instance {instance_idx}:",
        "\n=== Positive Contributions ==="
    ]

    for text, val in positive[:top_n]:
        result.append(f"({val:+.2f}) {text.strip()}")

    result.append("\n=== Negative Contributions ===")

    for text, val in negative[:top_n]:
        result.append(f"({val:+.2f}) {text.strip()}")

    return "\n".join(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)