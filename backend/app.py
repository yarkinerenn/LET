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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np
import shap
import io
from cryptography.fernet import Fernet
from groq import Groq
import pandas as pd
import tempfile
from datasets import load_dataset
import torch
load_dotenv()
# Store API key in .env
secret_key = os.getenv("SECRET_KEY")
secret_key = secret_key.encode()
cipher = Fernet(secret_key)
CLASSIFICATION_METHODS = ['llm', 'bert']
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/auth_app'
app.config['CORS_SUPPORTS_CREDENTIALS'] = True
app.config['SESSION_COOKIE_NAME'] = 'your_session_cookie_name'  # Optional, you can change the cookie name
CORS(app, origins=["http://localhost:3001"], supports_credentials=True)
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

@app.route('/api/classify/<dataset_id>', methods=['POST'])
@login_required
def classify_dataset(dataset_id):
    # Validate request data
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # Extract and validate parameters
        method = data.get('method')
        if method not in CLASSIFICATION_METHODS:
            return jsonify({"error": f"Invalid method. Must be one of: {CLASSIFICATION_METHODS}"}), 400

        provider = data.get('provider', 'openai')
        model_name = data.get('model', 'gpt-3.5-turbo')
        text_column = data.get('text_column', 'text')
        label_column = data.get('label_column','label')
        print(provider)
        # Get dataset metadata
        dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Load dataset with error handling
        try:
            df = pd.read_csv(dataset['filepath'])

            if text_column not in df.columns:
                return jsonify({"error": f"Text column '{text_column}' not found in dataset"}), 400
            if label_column and label_column not in df.columns:
                return jsonify({"error": f"Label column '{label_column}' not found in dataset"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500

        # Initialize classification client
        client = None
        if method == 'llm':
            if provider == 'openai':
                api_key = get_user_api_key_openai()
                if not api_key:
                    return jsonify({"error": "OpenAI API key not configured"}), 400
                client = OpenAI(api_key=api_key)
            elif provider == 'groq':
                api_key = get_user_api_key_groq()
                if not api_key:
                    return jsonify({"error": "Groq API key not configured"}), 400
                client = Groq(api_key=api_key)
            elif provider == 'deepseek':
                api_key = get_user_api_key_deepseek_api()
                if not api_key:
                    return jsonify({"error": "Deepseek API key not configured"}), 400
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            else:
                return jsonify({"error": "Invalid LLM provider"}), 400

        results = []
        stats = {
            "total": 0,
            "positive": 0,
            "negative": 0
        }

        # Process samples (limiting to first 100 for demo)
        df.head(100)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.head(100)
        samples = df.head(10).iterrows()
        for _, row in tqdm(samples, total=min(100, len(df)), desc="Classifying"):
            try:
                text = str(row[text_column])[:500]  # Truncate long texts

                if method == 'bert':
                    result = classifier(text)[0]
                    label = result['label'].upper()
                    score = result['score']
                else:
                    # LLM classification
                    if provider in ['openai', 'deepseek']:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "user",
                                "content": f"Classify this text's sentiment as only POSITIVE or NEGATIVE: {text}"
                            }],
                            max_tokens=10
                        )
                        label = response.choices[0].message.content.strip().upper()
                    else:  # groq
                        response = client.chat.completions.create(
                            messages=[{
                                "role": "user",
                                "content": f"Respond with only POSITIVE or NEGATIVE for this text: {text}"
                            }],
                            model=model_name,
                            max_tokens=10
                        )
                        label = response.choices[0].message.content.strip().upper()

                    # Normalize label
                    label = "POSITIVE" if "POS" in label else "NEGATIVE"
                    score = 1.0

                # Store result
                result_data = {
                    "text": text,
                    "label": label,
                    "score": score,
                    "original_data": row.to_dict()
                }

                if label_column:
                    result_data["actualLabel"] = str(row[label_column]).strip().upper()

                results.append(result_data)

                # Update stats
                stats["total"] += 1
                if label == "POSITIVE":
                    stats["positive"] += 1
                else:
                    stats["negative"] += 1

            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue

        # Calculate metrics if ground truth available
        if label_column:
            try:
                # Convert labels to integers (0/1)
                y_true = df[label_column].head(len(results)).astype(int)
                y_pred = [1 if r['label'] == 'POSITIVE' else 0 for r in results]

                # Calculate binary metrics
                stats["accuracy"] = accuracy_score(y_true, y_pred)
                stats["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                stats["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                stats["f1_score"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                stats["confusion_matrix"] = {
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp)
                }

                # Add to results for frontend display
                for i, result in enumerate(results):
                    result["actualLabel"] = int(y_true.iloc[i]) if i < len(y_true) else None

            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")

        # Store results
        classification_data = {
            "dataset_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id),
            "method": method,
            "provider": provider if method == 'llm' else None,
            "model": model_name if method == 'llm' else None,
            "results": results,
            "created_at": datetime.now(),
            "stats": stats
        }

        classification_id = mongo.db.classifications.insert_one(classification_data).inserted_id

        return jsonify({
            "message": "Classification completed",
            "classification_id": str(classification_id),
            "stats": stats,
            "sample_count": len(results)
        }), 200

    except Exception as e:
        print(f"Classification error: {str(e)}")
        return jsonify({
            "error": "Classification failed",
            "details": str(e)
        }), 500
@app.route('/api/classification/<classification_id>', methods=['GET'])
@login_required
def get_classification_details(classification_id):
    try:
        classification = mongo.db.classifications.find_one({
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(current_user.id)
        })

        if not classification:
            return jsonify({"error": "Classification not found"}), 404

        # Convert ObjectId and datetime
        classification["_id"] = str(classification["_id"])
        classification["dataset_id"] = str(classification["dataset_id"])
        classification["created_at"] = classification["created_at"].strftime("%Y-%m-%d %H:%M:%S")

        return jsonify(classification), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/classification/stats/<classification_id>', methods=['GET'])
@login_required
def get_classification_stats(classification_id):
    try:
        classification = mongo.db.classifications.find_one({
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(current_user.id)
        }, {"stats": 1, "method": 1, "model": 1, "provider": 1})

        if not classification:
            return jsonify({"error": "Classification not found"}), 404

        # Calculate additional metrics
        stats = classification.get("stats", {})
        stats.update({
            "accuracy": stats.get("accuracy", 0),
            "precision": stats.get("precision", 0),
            "recall": stats.get("recall", 0),
            "f1_score": stats.get("f1_score", 0)
        })

        return jsonify({
            "method": classification.get("method"),
            "model": classification.get("model"),
            "provider": classification.get("provider"),
            "stats": stats
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/classifications/<dataset_id>', methods=['GET'])
@login_required
def get_dataset_classifications(dataset_id):
    try:
        classifications = list(mongo.db.classifications.find({
            "dataset_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id)
        }, {
            "method": 1,
            "provider": 1,
            "model": 1,
            "created_at": 1,
            "stats": 1
        }).sort("created_at", -1))

        for cls in classifications:
            cls['_id'] = str(cls['_id'])
            cls['created_at'] = cls['created_at'].strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({"classifications": classifications}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_prediction/<classification_id>', methods=['DELETE'])
def delete_prediction(classification_id):
    try:
        # Delete classification from the database
        result = mongo.db.predictions.delete_one({'_id': ObjectId(classification_id)})

        if result.deleted_count == 0:
            return jsonify({"error": "Classification not found"}), 404

        return jsonify({"message": "Classification deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/delete_classification/<classification_id>', methods=['DELETE'])
@login_required
def delete_classification(classification_id):
    try:
        result = mongo.db.classifications.delete_one({
            '_id': ObjectId(classification_id),
            'user_id': ObjectId(current_user.id)  # Ensure user can only delete their own
        })

        if result.deleted_count == 0:
            return jsonify({"error": "Classification not found or unauthorized"}), 404

        return jsonify({"message": "Classification deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/import_hf_dataset", methods=["POST"])
@login_required
def import_hf_dataset():
    data = request.json
    hf_dataset_name = data.get("dataset_name")
    file_name = data.get("file_name", hf_dataset_name + ".csv")

    if not hf_dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(hf_dataset_name)
        df = dataset["train"].to_pandas()

        # Create file path within upload directory
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file_name)

        # Clean up existing file if needed
        if os.path.exists(filepath):
            os.remove(filepath)

        df.to_csv(filepath, index=False)

        # Save dataset metadata
        dataset_entry = {
            "filename": file_name,
            "filepath": filepath,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "HuggingFace",
            "user_id": ObjectId(current_user.id)
        }
        mongo.db.datasets.insert_one(dataset_entry)

        return jsonify({"message": "Dataset imported successfully"}), 201

    except Exception as e:
        # Clean up any temporary files on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        print(f"Error importing HF dataset: {str(e)}")
        return jsonify({"error": f"Failed to import dataset: {str(e)}"}), 500

# Updated delete endpoint with proper temp handling
@app.route("/api/delete_dataset/<dataset_id>", methods=["DELETE"])
@login_required
def delete_dataset(dataset_id):
    try:
        dataset = mongo.db.datasets.find_one({
            "_id": ObjectId(dataset_id),
        })

        if not dataset:
            return jsonify({"error": "Dataset not found or unauthorized"}), 404

        file_path = dataset.get("filepath")

        # Delete from MongoDB first
        result = mongo.db.datasets.delete_one({"_id": ObjectId(dataset_id)})

        if result.deleted_count == 0:
            return jsonify({"error": "Dataset not found"}), 404

        # File deletion logic
        if file_path and os.path.exists(file_path):
            upload_dir = os.path.abspath(app.config["UPLOAD_FOLDER"])
            file_abs_path = os.path.abspath(file_path)

            if not file_abs_path.startswith(upload_dir):
                app.logger.warning(f"Blocked attempt to delete file outside upload directory: {file_path}")
                return jsonify({
                    "message": "Dataset record deleted but file was preserved for security",
                    "warning": "File was outside protected directory"
                }), 200

            os.remove(file_path)
            app.logger.info(f"Deleted dataset file: {file_path}")

        return jsonify({"message": "Dataset and file deleted successfully"}), 200

    except Exception as e:
        app.logger.error(f"Delete error: {str(e)}")
        return jsonify({"error": f"Failed to delete dataset: {str(e)}"}), 500
# 📌 Upload Dataset
@app.route("/api/upload_dataset", methods=["POST"])
@login_required
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
        "source": "Local Upload",
        "user_id": ObjectId(current_user.id)
    }
    mongo.db.datasets.insert_one(dataset_entry)

    return jsonify({"message": "Dataset uploaded successfully"}), 201


# 📌 Retrieve Uploaded Datasets
@app.route("/api/datasets", methods=["GET"])
def get_datasets():
    datasets = list(mongo.db.datasets.find({}, {"_id": 1, "filename": 1, "uploaded_at": 1,'source':1}))
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
        self.deepseek_api = user_data.get('deepseek_api', '')
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
    deepseek_api_key = data.get("deepseek_api", "")


    # Encrypt the user-entered API keys
    encrypted_openai_api = encrypt_api_key(openai_api_key) if openai_api_key else ""
    encrypted_grok_api = encrypt_api_key(grok_api_key) if grok_api_key else ""
    encrypted_deepseek_api = encrypt_api_key(deepseek_api_key) if deepseek_api_key else ""


    user_data = {
            'username': data['username'],
            'email': data['email'],
            'password_hash': password_hash,
            'role': 'user',  # Default role
            'openai_api': encrypted_openai_api,  # Encrypted API key
            'grok_api': encrypted_grok_api,
            'deepseel_api': encrypted_deepseek_api# Encrypted API key
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
def get_user_api_key_deepseek_api():
    """Retrieve the user's OpenAI API key securely."""
    if not current_user.is_authenticated:
        return None

    # Fetch the OpenAI API key from MongoDB to avoid issues with Flask-Login session
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'deepseek_api': 1})
    print(current_user.username,'this is the user')

    if user_data and "deepseek_api" in user_data:
        return decrypt_api_key(user_data['deepseek_api'])  # Return decrypted API key
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
    deepseek_api_key = data.get("deepseek_api")

    update_fields = {}

    # Encrypt and update only if the user provided a new key
    if openai_api_key:
        update_fields["openai_api"] = encrypt_api_key(openai_api_key)

    if grok_api_key:
        update_fields["grok_api"] = encrypt_api_key(grok_api_key)

    if deepseek_api_key:
        update_fields["deepseek_api"] = encrypt_api_key(deepseek_api_key)

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
        print('very true')
        return jsonify({
            "authenticated": True,
            "user": {
                "id": current_user.id,
                "username": current_user.username
            }
        })
    print('very false')
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
        print(result['label'])

        return jsonify({
            "id": str(prediction.inserted_id),
            "label": result['label'],
            "score": result['score']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyzewithllm', methods=['POST'])
@login_required
def analyze_text_with_llm():
    try:
        # Get the text from the request
        data = request.json
        text = data.get('text', '')
        provider=data.get('provider')
        model=data.get('model')

        if provider=='openai':
            openai_api_key = get_user_api_key_openai()
            client = OpenAI(api_key=openai_api_key)


            if len(text) < 3:
                    return jsonify({"error": "Text must be at least 3 characters"}), 400

            # Send the text to OpenAI for sentiment analysis
            prompt = f"Classify the sentiment of the following text as either positive or negative:\n{text}"

            # Call OpenAI GPT-3/4 model to analyze the sentiment
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract the sentiment result (positive or negative)
            sentiment =  response.choices[0].message.content

            # Ensure the sentiment is either positive or negative
            if sentiment not in ["positive", "negative"]:
                return jsonify({"error": "Invalid sentiment response from LLM."}), 400
        elif provider=='groq':
            api= get_user_api_key_groq()

            client = Groq(
                api_key=api,
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content":f"Classify the sentiment of the following text as either positive or negative only one word:\n{text}",
                    }
                ],
                model=model,
            )

            sentiment= chat_completion.choices[0].message.content

            if 'positive' in sentiment.lower():
                sentiment = "POSITIVE"
            else:
                sentiment = "NEGATIVE"

            if sentiment not in ["POSITIVE", "NEGATIVE"]:
                return jsonify({"error": "Invalid sentiment response from LLM."}), 400
        elif provider=='deepseek':
            deepseek_api_key = get_user_api_key_deepseek_api()
            client = OpenAI(api_key=deepseek_api_key)


            if len(text) < 3:
                return jsonify({"error": "Text must be at least 3 characters"}), 400

            # Send the text to OpenAI for sentiment analysis
            prompt = f"Classify the sentiment of the following text as either positive or negative:\n{text}"

            # Call OpenAI GPT-3/4 model to analyze the sentiment
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            # Extract the sentiment result (positive or negative)
            sentiment =  response.choices[0].message.content

            # Ensure the sentiment is either positive or negative
            if sentiment not in ["positive", "negative"]:
                return jsonify({"error": "Invalid sentiment response from LLM."}), 400


        # Store in MongoDB
        prediction = mongo.db.predictions.insert_one({
            "text": text,
            "label": sentiment,
            "score": 1,  # Assuming the LLM is fully confident here (optional)
            "timestamp": datetime.now()
        })

        return jsonify({
            "id": str(prediction.inserted_id),
            "label": sentiment,
            "score": 1  # Use actual model score if available
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/explain', methods=['POST'])
@login_required
def explain_prediction():
    try:
        data = request.json
        prediction_id = data.get('prediction_id','fromdata')
        predictedlabel=data.get('predictedlabel')
        truelabel=data.get('truelabel')
        confidence=data.get('confidence')
        text = data.get('text'  )
        provider=data.get('provider')
        model=data.get('model')
        explainer_type = data.get('explainer_type', 'llm')

        if prediction_id=='fromdata':
            if not text:
                return jsonify({"error": "Missing text"}), 400

            if explainer_type == 'shap':
                explanation_data,top_words = generate_shap_explanation(text, predictedlabel)
                return jsonify({'explanation': explanation_data,'explainer_type': explainer_type , 'top_words': top_words})
            else:
                explanation_text = generate_llm_explanationofdataset(text, predictedlabel,truelabel, confidence,provider,model)
                return jsonify({"explanation": explanation_text,'explainer_type': explainer_type})


        else:

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
def generate_llm_explanationofdataset(text, label,truelabel, score,provider,model):
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

            return chat_completion.choices[0].message.content



    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"
@app.route('/api/explanation/<classification_id>/<result_id>', methods=['GET'])
@login_required
def get_explanation(classification_id, result_id):
    try:
        classification = mongo.db.classifications.find_one({
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(current_user.id)
        })

        if not classification:
            return jsonify({"error": "Classification not found"}), 404

        result = classification['results'][int(result_id)]

        return jsonify({
            "text": result['text'],
            "prediction": result['label'],
            "confidence": result['score'],
            "actualLabel": result.get('actualLabel'),
            "provider": classification.get('provider'),  # Added provider
            "model": classification.get('model'),        # Added model
            # Removed the pre-generated explanation and important words
            # These will be generated on-demand by the frontend
        })

    except IndexError:
        return jsonify({"error": "Result not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
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