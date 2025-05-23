import requests
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
import traceback  # EN ÜSTE EKLE

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
def classify_with_chunks(text, classifier, tokenizer, max_length=512, stride=256):
    """use it if the context windows of the BERT is not big enoguh for a dataset entry"""
    inputs = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=True,  # bu eksikti
        return_tensors="pt"
    )

    scores = []
    labels = []
    for chunk in inputs['input_ids']:
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        try:
            result = classifier(decoded)[0]
            scores.append(result['score'])
            labels.append(result['label'].upper())
        except:
            continue

    if not scores:
        return "NEGATIVE", 0.0

    # Majority voting or average
    avg_score = sum(scores) / len(scores)
    final_label = max(set(labels), key=labels.count)  # majority vote

    return final_label, avg_score
@app.route("/api/dataset/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    """Get the uploaded dataset"""

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
    """Classify the whole dataset using either BERT or generative AI """
    # Validate request data
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # Extract and validate parameters
        method = data.get('method')
        if method not in CLASSIFICATION_METHODS:
            return jsonify({"error": f"Invalid method. Must be one of: {CLASSIFICATION_METHODS}"}), 400

        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        provider = user_doc.get('preferred_provider', 'openai')
        model_name = user_doc.get('preferred_model', 'gpt-3.5-turbo')
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
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        samples = df.head(100).iterrows()
        for _, row in tqdm(samples, total=min(100, len(df)), desc="Classifying"):
            try:
                text = str(row[text_column])# Truncate long texts

                if method == 'bert':
                    label, score = classify_with_chunks(text, classifier, tokenizer)
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
                    "original_data": row.to_dict(),
                    "llm_explanation": "",
                    "shap_plot_explanation":"",
                    "shapwithllm_explanation": "",

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
def save_explanation_to_db(classification_id, user_id, result_id, explanation_type, content):
    """Save the explanations generated by the user """
    if explanation_type == 'llm':
        update_field = f'results.{result_id}.llm_explanation'
        print(update_field)

    elif explanation_type == 'shap_plot':

        update_field = f'results.{result_id}.shap_plot_explanation'
        print(update_field)
    elif explanation_type == 'shapwithllm':
        update_field = f'results.{result_id}.shapwithllm_explanation'
        print(update_field)

    result = mongo.db.classifications.update_one(
        {
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(user_id),
            f"results.{result_id}": {"$exists": True}
        },
        {
            "$set": {update_field: content}
        }
    )
    return result.modified_count > 0
@app.route('/api/save-explanation', methods=['POST'])
@login_required
def save_explanation():
    """Save the explanations generated by the user """

    try:
        data = request.get_json()
        print(data)
        classification_id = data['classification_id']
        result_id = int(data['result_id'])
        explanation_type = data['explanation_type']
        content = data['content']

        # Map explanation types to database fields
        if explanation_type=='llm':
            field_map = {
                'llm': f'results.{result_id}.llm_explanation',
            }
        elif explanation_type=='shap_plot':
            field_map = {
                'shap_plot': f'results.{result_id}.shap_plot_explanation',
            }

        else:
            field_map = {
                'shapwithllm': f'results.{result_id}.shapwithllm_explanation'
            }

        update_field = field_map[explanation_type]

        result = mongo.db.classifications.update_one(
            {
                "_id": ObjectId(classification_id),
                "user_id": ObjectId(current_user.id),
                f"results.{result_id}": {"$exists": True}
            },
            {
                "$set": {update_field: content}
            }
        )
        if result.modified_count == 0:
            return jsonify({"error": "Failed to save explanation"}), 400

        return jsonify({"message": "Explanation saved successfully"}), 200

    except Exception as e:
        traceback.print_exc()  # <-- HATA BURADA GÖRÜNÜR

        return jsonify({"error": str(e)}), 500
@app.route('/api/track-selection', methods=['POST'])
@login_required
def track_selection():
    """track the best explanation selection """
    try:
        user_id=ObjectId(current_user.id)
        data = request.get_json()
        classification_id = data.get('classificationId')
        result_id = data.get('resultId')
        selected_type = data.get('selectedType')
        timestamp = data.get('timestamp', datetime.utcnow().isoformat())

        if not all([classification_id, result_id, selected_type]):
            return jsonify({"error": "Missing required fields"}), 400

        selection_data = {
            "user_id": user_id,
            "classification_id": classification_id,
            "result_id": result_id,
            "selected_type": selected_type,
            "timestamp": timestamp
        }

        mongo.db.selections.insert_one(selection_data)

        return jsonify({"message": "Selection tracked successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/classification/<classification_id>', methods=['GET'])
@login_required
def get_classification_details(classification_id):
    """Get the classification details to see previous classification on the detailed dataset view page  """
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
    """Get the classification statistics """
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
@app.route('/api/predictions/update_prediction_label/<prediction_id>', methods=['POST'])
def update_prediction_label(prediction_id):
    """User can label their own data entry"""
    user_label = request.json.get("user_label")

    if not user_label:
        return {"error": "Missing 'user_label' in request"}, 400

    result = mongo.db.predictions.update_one(
        {"_id": ObjectId(prediction_id)},
        {"$set": {"user_label": user_label}}
    )

    if result.matched_count == 0:
        return {"error": "Prediction not found"}, 404

    return {"message": "Label updated successfully"}, 200



@app.route('/api/delete_prediction/<classification_id>', methods=['DELETE'])
def delete_prediction(classification_id):
    """Delete a prediction that is made in the dashboard page"""
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
    """Delete a clasification that is made for a whole dataset"""
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
    """Import a dataset from hugginface to the database"""
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
    """Delete an imported dataset form the database"""
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
    """Upload a local dataset"""
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
    """Get the datasets that is on the database uploaded by the user"""
    user_id = current_user.id
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    datasets = list(mongo.db.datasets.find({"user_id": user_id}, {"_id": 1, "filename": 1, "uploaded_at": 1,'source':1}))
    for dataset in datasets:
        dataset["_id"] = str(dataset["_id"])
    return jsonify({"datasets": datasets})


class User(UserMixin):
    """User class"""
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']
        self.role = user_data.get('role', 'user')  # Default to 'user'
        self.openai_api = user_data.get('openai_api', '')  # Ensure openai_api is loaded
        self.grok_api = user_data.get('grok_api', '')  # Ensure grok_api is loaded
        self.deepseek_api = user_data.get('deepseek_api', '')
        self.preferred_provider = user_data.get('preferred_provider', 'openai')
        self.preferred_model = user_data.get('preferred_model', 'gpt-3.5-turbo')
        self.preferred_providerex = user_data.get('preferred_providerex', 'openai')
        self.preferred_modelex = user_data.get('preferred_modelex', 'gpt-3.5-turbo')
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
        'role': 'user',
        'openai_api': encrypted_openai_api,
        'grok_api': encrypted_grok_api,
        'deepseek_api': encrypted_deepseek_api,
        'preferred_provider': data.get('preferred_provider', 'openai'),
        'preferred_model': data.get('preferred_model', 'gpt-3.5-turbo'),
        'preferred_providerex': data.get('preferred_providerex', 'openai'),
        'preferred_modelex': data.get('preferred_modelex', 'gpt-3.5-turbo')
    }

    result = mongo.db.users.insert_one(user_data)

    return jsonify({
        "message": "User created successfully",
        "id": str(result.inserted_id)
    }), 201
@app.route('/api/settings/update_preferred_classification', methods=['POST'])
@login_required
def update_preferred_classification():
    """Save the user's preferred generative AI settings for the classification"""
    data = request.json
    preferred_provider = data.get('preferred_provider', 'openai')
    preferred_model = data.get('preferred_model', 'gpt-3.5-turbo')

    mongo.db.users.update_one(
        {'_id': ObjectId(current_user.id)},
        {'$set': {
            'preferred_provider': preferred_provider,
            'preferred_model': preferred_model
        }}
    )

    return jsonify({"message": "Classification preferences updated successfully"}), 200
@app.route('/api/settings/update_preferred_explanation', methods=['POST'])
@login_required
def update_preferred_explanation():
    """Save the user's preferred generative AI settings for the explanations"""
    data = request.json
    preferred_providerex = data.get('preferred_providerex', 'openai')
    preferred_modelex = data.get('preferred_modelex', 'gpt-3.5-turbo')

    mongo.db.users.update_one(
        {'_id': ObjectId(current_user.id)},
        {'$set': {
            'preferred_providerex': preferred_providerex,
            'preferred_modelex': preferred_modelex
        }}
    )

    return jsonify({"message": "Explanation preferences updated successfully"}), 200
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
    """Login the user"""

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
    """Update the generative AI api keys of the user"""
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
    """Authentication for the @login_required functions"""
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
    """Classify e signgle instanve with BERT then insert it to database"""

    try:
        data = request.json
        text = data.get('text', '')
        print(text)

        if len(text) < 3:
            return jsonify({"error": "Text must be at least 3 characters"}), 400

        # Classify text
        result = classifier(text)[0]
        user_id = current_user.id
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        # Store in MongoDB
        prediction = mongo.db.predictions.insert_one({
            'model': 'bert',
            "user_id": user_id,
            "text": text,
            "label": result['label'],
            "score": result['score'],
            "timestamp": datetime.now(),
            "user_label": "",
        })

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
    """Classify a single entry with generative ai model of users choice"""

    try:
        # Get the text from the request
        data = request.json
        text = data.get('text', '')
        # Get user's preferred classification provider and model
        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        provider = user_doc.get('preferred_provider', 'openai')
        model = user_doc.get('preferred_model', 'gpt-3.5-turbo')


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

        user_id = current_user.id
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        # Store in MongoDB
        prediction = mongo.db.predictions.insert_one({
            'model':"llm",
            "user_id": user_id,
            "text": text,
            "label": sentiment,
            "score": 1,  # Assuming the LLM is fully confident here (optional)
            "timestamp": datetime.now(),
            "user_label": "",
        })

        return jsonify({
            "id": str(prediction.inserted_id),
            "label": sentiment,
            "score": 1  # Use actual model score if available
        })

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500
@app.route('/api/explain', methods=['POST'])
@login_required
def explain_prediction():
    """Generate explanations shap or generative AI explanations for classification"""

    try:
        data = request.json
        prediction_id = data.get('prediction_id','fromdata')
        classificationId=data.get('classificationId')
        resultId=data.get('resultId')
        predictedlabel=data.get('predictedlabel')

        truelabel=data.get('truelabel')
        confidence=data.get('confidence')
        text = data.get('text'  )
        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        provider = user_doc.get('preferred_providerex', 'openai')
        model = user_doc.get('preferred_modelex', 'gpt-3.5-turbo')
        explainer_type = data.get('explainer_type', 'llm')

        if prediction_id=='fromdata':
            if not text:
                return jsonify({"error": "Missing text"}), 400

            if explainer_type == 'shap':
                explanation_data,top_words = generate_shap_explanation(text, predictedlabel)
                # Save SHAP plot
                save_explanation_to_db(classificationId,current_user.id,resultId,'shap_plot',explanation_data)

                return jsonify({'explanation': explanation_data,'explainer_type': explainer_type , 'top_words': top_words})
            else:
                explanation_text = generate_llm_explanationofdataset(text, predictedlabel,truelabel, confidence,provider,model)
                save_explanation_to_db(classificationId,current_user.id,resultId,'llm',explanation_text)

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
    """generate explanations with generative AI"""

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
    """Generete generative AI explanation of singel instances in the dataset"""


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
        # Map label to class index, but make sure we check the prediction first
        class_index = 1 if label == 'POSITIVE' else 0

        # Generate the explanation
        pmodel = shap.models.TransformersPipeline(classifier, rescale_to_logits=True)
        _ = pmodel([input_text])  # Trigger internal stuff
        explainer2 = shap.Explainer(pmodel)
        shap_values2 = explainer2([input_text])

        # Check prediction: If prediction is POSITIVE, we need class_index = 0
        # because shap is showing the opposing class's influence
        predicted_class = np.argmax(shap_values2.values.sum(axis=1))
        if predicted_class == 1:  # If predicted POSITIVE, swap the class_index
            class_index = 0  # For POSITIVE predictions, we need to use the negative class index

        output_str = get_top_phrases(shap_values2, top_n=10)
        plot = shap.plots.text(shap_values2[:, :, class_index], display=False)

        return plot, output_str
    except Exception as e:
        print(f"Error in SHAP explanation: {str(e)}")
        return None, f"Could not generate SHAP explanation: {str(e)}"
@app.route('/api/explain_withshap', methods=['POST'])
def generate_llm_explanation_of_shap():
    """Generate generative AI explanation with prompting SHAP values"""


    try:
        data = request.json
        shapwords=data.get('shapwords')
        classificationId=data.get('classificationId')
        resultId=data.get('resultId')
        text = data.get('text')
        prediction_id = data.get('prediction_id')
        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        provider = user_doc.get('preferred_providerex', 'openai')
        model = user_doc.get('preferred_modelex', 'gpt-3.5-turbo')
        explainer_type = data.get('explainer_type', 'llm')
        if prediction_id:

            if not prediction_id or not text:
                return jsonify({"error": "Missing prediction_id or text"}), 400

            prediction = mongo.db.predictions.find_one({"_id": ObjectId(prediction_id)})
            if not prediction:
                return jsonify({"error": "Prediction not found"}), 404

            label=prediction['label']
            score=prediction['score']
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

                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',explanation)

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

                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',chat_completion.choices[0].message.content)


                return chat_completion.choices[0].message.content

        else:
            if not text:
                return jsonify({"error": "Missing prediction_id or text"}), 400

            label=data.get('label')
            score=data.get('confidence')
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
                requests.post('http://localhost:5000/api/save-explanation', json={
                    'classification_id': classificationId,
                    'result_id': int(resultId),
                    'explanation_type': 'shapwithllm',
                    'content': explanation
                })
                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',explanation)

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


                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',chat_completion.choices[0].message.content)


                return chat_completion.choices[0].message.content



    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {str(e)}"
@app.route('/api/classificationentry/<classification_id>/<result_id>', methods=['GET'])
@login_required
def get_classificationentry(classification_id, result_id):
    """Get a single entry of a classified dataset"""
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
            "llm_explanation": result.get('llm_explanation', ''),
            "shap_plot": result.get('shap_plot_explanation', ''),
            "shapwithllm": result.get('shapwithllm_explanation', ''),
            "provider": classification.get('provider'),
            "model": classification.get('model')
        })

    except IndexError:
        return jsonify({"error": "Result not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/predictions/<prediction_id>', methods=['GET'])
@login_required
def get_prediction_by_id(prediction_id):
    """Get a specific prediction by its ObjectId"""
    try:
        user_id = current_user.id
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)

        prediction = mongo.db.predictions.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": user_id
        })

        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        result = {
            'model':prediction["model"],
            "id": str(prediction["_id"]),
            "text": prediction["text"],
            "label": prediction["label"],
            "confidence": prediction["score"],
            "timestamp": prediction["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({"classification": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/api/predictions', methods=['GET'])
@login_required
def get_predictions():
    """Get the classification made from the dashboard independent of any dataset"""
    try:
        user_id = current_user.id
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        predictions = (
            mongo.db.predictions
            .find({"user_id": user_id})
            .sort("timestamp", -1)
            .limit(50)
        )
        # Convert ObjectId to string and prepare JSON response
        results = []
        for prediction in predictions:
            results.append({
                "model":prediction["model"],
                "id": str(prediction["_id"]),
                "text": prediction["text"],
                "label": prediction["label"],
                "score": prediction["score"],  # Convert score to percentage
                "timestamp": prediction["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            })

        print(results,'these are the results')

        return jsonify({"classifications": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
def get_top_phrases(shap_values_obj, instance_idx=0, class_idx=1, top_n=5):
    """From the shapley value extract the most affecting word for the classification"""
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