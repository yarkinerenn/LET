from flask import Flask, jsonify, request,send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_cors import CORS
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from transformers import pipeline,AutoModelForSequenceClassification,AutoTokenizer
from dotenv import load_dotenv
from openai import OpenAI
import os
import numpy as np
import shap
import io
from cryptography.fernet import Fernet
from groq import Groq
import pandas as pd

import torch
load_dotenv()
# Store API key in .env
secret_key = os.getenv("SECRET_KEY")
secret_key = secret_key.encode()
cipher = Fernet(secret_key)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/auth_app'
app.config['CORS_SUPPORTS_CREDENTIALS'] = True

CORS(app, origins=["http://localhost:3000"], supports_credentials=True)
mongo = PyMongo(app)
login_manager = LoginManager()
login_manager.init_app(app)
app.config["UPLOAD_FOLDER"] = "uploads"
CORS(app, supports_credentials=True)

# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

@app.route("/api/dataset/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404

    filepath = dataset["filepath"]

    try:
        df = pd.read_csv(filepath)
        data_preview = df.head(20).to_dict(orient="records")  # Return first 20 rows
        return jsonify({"filename": dataset["filename"], "data": data_preview})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

# 📌 Delete Dataset by ID
@app.route("/api/delete_dataset/<dataset_id>", methods=["DELETE"])
def delete_dataset(dataset_id):
    result = mongo.db.datasets.delete_one({"_id": ObjectId(dataset_id)})

    if result.deleted_count == 0:
        return jsonify({"error": "Dataset not found"}), 404

    return jsonify({"message": "Dataset deleted successfully"}), 200
# 📌 Upload Dataset
@app.route("/api/upload_dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Save dataset metadata in MongoDB
    dataset_entry = {
        "filename": filename,
        "filepath": filepath,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    mongo.db.datasets.insert_one(dataset_entry)

    return jsonify({"message": "Dataset uploaded successfully"}), 201


# 📌 Retrieve Uploaded Datasets
@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    datasets = list(mongo.db.datasets.find({}, {"_id": 1, "filename": 1, "uploaded_at": 1}))
    for dataset in datasets:
        dataset["_id"] = str(dataset["_id"])
    return jsonify({"datasets": datasets})


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']
        self.role = user_data.get('role', 'user')  # Default to 'user'
        self.openai_api = user_data.get('openai_api', '')  # Ensure openai_api is loaded
        self.grok_api = user_data.get('grok_api', '')  # Ensure grok_api is loaded

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

    # Check if email already exists
    if mongo.db.users.find_one({'email': data['email']}):
        return jsonify({"error": "Email already exists"}), 400

    # Hash the password
    password_hash = generate_password_hash(data['password'])

    # Encrypt the API keys entered by the user
    openai_api_key = data.get("openai_api", "")
    grok_api_key = data.get("grok_api", "")

    # Encrypt the user-entered API keys
    encrypted_openai_api = encrypt_api_key(openai_api_key) if openai_api_key else ""
    encrypted_grok_api = encrypt_api_key(grok_api_key) if grok_api_key else ""

    user_data = {
        'username': data['username'],
        'email': data['email'],
        'password_hash': password_hash,
        'role': 'user',  # Default role
        'openai_api': encrypted_openai_api,  # Encrypted API key
        'grok_api': encrypted_grok_api       # Encrypted API key
    }

    result = mongo.db.users.insert_one(user_data)

    return jsonify({
        "message": "User created successfully",
        "id": str(result.inserted_id)
    }), 201
def get_user_api_key_openai():
    """Retrieve the user's OpenAI API key securely."""
    if not current_user.is_authenticated:
        return None

    # Fetch the OpenAI API key from MongoDB to avoid issues with Flask-Login session
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'openai_api': 1})
    print(current_user.username,'this is the user')

    if user_data and "openai_api" in user_data:
        return decrypt_api_key(user_data['openai_api'])  # Return decrypted API key
    return None
def get_user_api_key_groq():
    """Retrieve the user's Groq API key securely."""
    if not current_user.is_authenticated:
        return None

    # Fetch the OpenAI API key from MongoDB to avoid issues with Flask-Login session
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'grok_api': 1})
    print(current_user.username,'this is the user')

    if user_data and "grok_api" in user_data:
        return decrypt_api_key(user_data['grok_api'])  # Return decrypted API key
    return None

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
@app.route('/api/settings/update_api_keys', methods=['POST'])
@login_required
def update_api_keys():
    data = request.json
    openai_api_key = data.get("openai_api")
    grok_api_key = data.get("grok_api")

    update_fields = {}

    # Encrypt and update only if the user provided a new key
    if openai_api_key:
        update_fields["openai_api"] = encrypt_api_key(openai_api_key)

    if grok_api_key:
        update_fields["grok_api"] = encrypt_api_key(grok_api_key)

    # Only update if there's something to change
    if update_fields:
        mongo.db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$set": update_fields}
        )

    return jsonify({"message": "API keys updated successfully"})

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
@login_required
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
@login_required
def explain_prediction():
    try:
        print("Current User:", current_user)  # Log the current user
        data = request.json
        prediction_id = data.get('prediction_id')
        text = data.get('text'  )
        provider=data.get('provider')
        model=data.get('model')
        explainer_type = data.get('explainer_type', 'llm')

        if not prediction_id or not text:
            return jsonify({"error": "Missing prediction_id or text"}), 400

        prediction = mongo.db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        if explainer_type == 'shap':
            explanation_data,top_words = generate_shap_explanation(text, prediction['label'])
            return jsonify({'explanation': explanation_data,'explainer_type': explainer_type , 'top_words': top_words})
        else:
            explanation_text = generate_llm_explanation(text, prediction['label'], prediction['score'],provider,model)
            return jsonify({"explanation": explanation_text,'explainer_type': explainer_type})

    except Exception as e:
        print(f"Error generating explanation: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_llm_explanation(text, label, score,provider,model):
    try:
        if provider == 'openai':

            openai_api_key = get_user_api_key_openai()

            if not openai_api_key:
                return "Error: No OpenAI API key found for this user."

            client = OpenAI(api_key=openai_api_key)

            prompt = f"""
                Explain this sentiment analysis result in simple terms:
                
                Text: {text}
                Sentiment: {label} ({score}% confidence)
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            explanation = response.choices[0].message.content
            return explanation
        else:
            api= get_user_api_key_groq()

            client = Groq(
                api_key=api,
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content":f"""
                Explain this sentiment analysis result in simple terms:
                
                Text: {text}
                Sentiment: {label} ({score}% confidence)
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """,
                    }
                ],
                model=model,
            )
            return chat_completion.choices[0].message.content



    except Exception as e:
            print(f"Error: {e}")
            return f"Error: {str(e)}"

def generate_shap_explanation(input_text, label):
    """Generate an explanation using SHAP values"""
    try:
        pmodel = shap.models.TransformersPipeline(classifier, rescale_to_logits=True)
        pmodel([input_text])
        explainer2 = shap.Explainer(pmodel)
        shap_values2 = explainer2([input_text])

        output_str = get_top_phrases(shap_values2, top_n=10)
        print(output_str)
        plot=shap.plots.text(shap_values2[:, :, 1],display=False)

        return plot,output_str
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error in SHAP explanation: {str(e)}")
        return f"Could not generate SHAP explanation: {str(e)}"

@app.route('/api/explain_withshap', methods=['POST'])
def generate_llm_explanation_of_shap():
    try:
        data = request.json
        text = data.get('text')
        prediction_id = data.get('prediction_id')
        model=data.get('model')
        provider=data.get('provider')
        explainer_type = data.get('explainer_type', 'llm')

        if not prediction_id or not text:
            return jsonify({"error": "Missing prediction_id or text"}), 400

        prediction = mongo.db.predictions.find_one({"_id": ObjectId(prediction_id)})
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        label=prediction['label']
        score=prediction['score']
        shapwords=data.get('shapstring')
        if provider == 'openai':

            openai_api_key = get_user_api_key_openai()

            if not openai_api_key:
                return "Error: No OpenAI API key found for this user."

            client = OpenAI(api_key=openai_api_key)

            prompt = f"""
                Explain this sentiment analysis result in simple terms with most affecting words provided by SHAP:
                
                Text: {text}
                Sentiment: {label} ({score}% confidence)
                
                shap: 
                
                {shapwords}
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """
            print(shapwords,'these are the values')

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            explanation = response.choices[0].message.content
            return explanation
        else:
            api= get_user_api_key_groq()

            client = Groq(
                api_key=api,
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content":f"""
                Explain this sentiment analysis result in simple terms with most affecting words provided by SHAP:
                
                Text: {text}
                Sentiment: {label} ({score}% confidence)
                
                shap: 

                {shapwords}
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """,
                    }
                ],
                model=model,
            )
            print(shapwords,'these are the values')

            return chat_completion.choices[0].message.content



    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"

@app.route('/api/classifications', methods=['GET'])
@login_required
def get_classifications():
    try:
        # Retrieve last 10 classifications (modify limit as needed)
        predictions = mongo.db.predictions.find().sort("timestamp", -1).limit(50)

        # Convert ObjectId to string and prepare JSON response
        results = []
        for prediction in predictions:
            results.append({
                "id": str(prediction["_id"]),
                "text": prediction["text"],
                "label": prediction["label"],
                "score": prediction["score"],  # Convert score to percentage
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

def encrypt_api_key(api_key: str) -> str:
    """Encrypts the API key before storing it in the database."""
    return cipher.encrypt(api_key.encode()).decode()
def decrypt_api_key(encrypted_api_key: str) -> str:
    """Decrypts an API key when retrieving from the database."""
    return cipher.decrypt(encrypted_api_key.encode()).decode()

if __name__ == '__main__':
    app.run(port=5000, debug=True)