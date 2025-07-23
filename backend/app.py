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
from LExT.metrics.faithfulness import faithfulness
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np
from langchain_community.llms import Ollama
import traceback  # EN ÜSTE EKLE
import ast
import shap
import re
from cryptography.fernet import Fernet
from groq import Groq
import pandas as pd
import tempfile
from datasets import load_dataset
import torch
from sklearn.model_selection import train_test_split
from LExT.metrics.trustworthiness import lext
import ollama

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
@app.route('/api/classification/<classification_id>/add_explanation_models', methods=['POST'])
@login_required
def add_explanationmodels_to_classficication(classification_id):
    data = request.get_json()
    explanation_models = data.get('explanation_models', [])
    mongo.db.classifications.update_one(
            {
                "_id": ObjectId(classification_id),
                "user_id": ObjectId(current_user.id)  # Ensure user owns this classification
            },
            {
                "$set": {
                    "explanation_models": explanation_models,
                    "updated_at": datetime.now()
                }
            }
        )

    return jsonify({
        "message": "Explanation models added successfully",
        "classification_id": classification_id
    }), 200


@app.route('/api/trustworthiness', methods=['POST'])
@login_required
def trustworthiness_endpoint():
    # Parse POST data
    data = request.json
    print(data)


    # Extract fields from request
    question = data.get("ground_question")
    ground_explanation = data.get("ground_explanation")
    ground_label = data.get("ground_label")
    explanation = data.get("predicted_explanation")
    label = data.get("predicted_label")
    context = data.get("ground_context", None)    # Optional: use if your faithfulness function supports it
    context = extract_context_explanation(context)
    groq = get_user_api_key_groq()                     # Or however your code names these
    target_model = data.get("target_model")
    provider=data.get("provider")
    if provider == "openrouter":
        api=get_user_api_key_openrouter()
    elif provider == "openai":
        api=get_user_api_key_openai()
    elif provider == "groq":
        api=get_user_api_key_groq()
    else:
        api='api'
    ner_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

    # Prepare row_reference for tracking all info/results
    row_reference = {
        "ground_question": question,
        "ground_explanation": ground_explanation,
        "ground_label": ground_label,
        "predicted_explanation": explanation,
        "predicted_label": label,
        "ground_context": context,
    }

    # Compute lext score
    score = lext(context, question,ground_explanation, ground_label, target_model, groq,provider,api,ner_pipe,  row_reference
    )

    # Return as JSON (including the whole row_reference if you want details)
    print(row_reference)
    # Extract Plausibility Metrics
    plausibility_metrics = {
        "iterative_stability": row_reference.get("iterative_stability"),
        "paraphrase_stability": row_reference.get("paraphrase_stability"),
        "consistency": row_reference.get("consistency"),
        "plausibility": row_reference.get("plausibility")
    }

    # Extract Faithfulness Metrics
    faithfulness_metrics = {
        "qag_score": row_reference.get("qag_score"),
        "counterfactual": row_reference.get("counterfactual_scaled"),  # or just "counterfactual"
        "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
        "faithfulness": row_reference.get("faithfulness")
    }

    # Trustworthiness
    trustworthiness_score = row_reference.get("trustworthiness")
    metrics={"plausibility_metrics": plausibility_metrics,"faithfulness_metrics": faithfulness_metrics,"trustworthiness_score": trustworthiness_score}
    return jsonify({
        "trustworthiness_score": score,
    })

@app.route('/api/faithfulness', methods=['POST'])
@login_required
def faithfulness_endpoint():
    # Parse POST data
    data = request.json
    print(data)


    # Extract fields from request
    question = data.get("ground_question")
    ground_explanation = data.get("ground_explanation")
    ground_label = data.get("ground_label")
    explanation = data.get("predicted_explanation")
    label = data.get("predicted_label")
    context = data.get("ground_context", None)    # Optional: use if your faithfulness function supports it
    if context:
        context = extract_context_explanation(context)
    groq = get_user_api_key_groq()                     # Or however your code names these
    target_model = 'llama3:8b'

    # Prepare row_reference for tracking all info/results
    row_reference = {
        "ground_question": question,
        "ground_explanation": ground_explanation,
        "ground_label": ground_label,
        "predicted_explanation": explanation,
        "predicted_label": label,
        "ground_context": context,
    }

    # Compute faithfulness score
    # Adjust the arguments below to match your actual faithfulness() function's signature!
    score = faithfulness(
        explanation, label, question, ground_label, context, groq, target_model,provider,api, row_reference
    )
    print(row_reference)

    # Return as JSON (including the whole row_reference if you want details)
    return jsonify({
        "faithfulness_score": score,
        "details": row_reference
    })


CLASSIFICATION_METHODS = ['bert', 'llm']

@app.route('/api/classify/<dataset_id>', methods=['POST'])
@login_required
def classify_dataset(dataset_id):
    """Classify the whole dataset (sentiment, legal/casehold, etc) using BERT or generative AI """
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # --- Extract parameters ---
        method = data.get('method')
        data_type = data.get('dataType', 'sentiment')
        if data_type == 'legal':
            text_column = data.get('text_column', 'citing_prompt')
            label_column = data.get('label_column', 'label')
        elif data_type == 'sentiment':
            text_column = data.get('text_column', 'text')
            label_column = data.get('label_column', 'label')
        else:
            text_column = "question"
            label_column = "final_decision"
        print(text_column, label_column)
        if method not in CLASSIFICATION_METHODS:
            return jsonify({"error": f"Invalid method. Must be one of: {CLASSIFICATION_METHODS}"}), 400

        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        provider = user_doc.get('preferred_provider', 'openai')
        model_name = user_doc.get('preferred_model', 'gpt-3.5-turbo')

        dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # --- Load dataset ---
        try:
            df = pd.read_csv(dataset['filepath'])

            if text_column not in df.columns:
                return jsonify({"error": f"Text column '{text_column}' not found in dataset"}), 400
            if label_column and label_column not in df.columns:
                return jsonify({"error": f"Label column '{label_column}' not found in dataset"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500

        # --- Init client ---
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

        # --- Process rows ---
        results = []
        stats = {"total": 0}
        if data_type == "sentiment":
            stats.update({"positive": 0, "negative": 0})
        elif data_type == "legal":
            stats.update({"correct": 0, "incorrect": 0})
        elif data_type == "medical":
            stats.update({"correct": 0, "incorrect": 0})

        # For demo/sample, limit to first 50
        sample_size = 20  # or whatever
        if label_column in df.columns:
            # Make sure you have enough samples in each class!
            df_sampled, _ = train_test_split(
                df,
                train_size=sample_size,
                stratify=df[label_column],
                random_state=42,
            )
        else:
            # fallback to random if label column not found
            df_sampled = df.sample(n=sample_size, random_state=42)

        samples = df_sampled.iterrows()

        def parse_casehold_label(content):
            content = content.strip()
            # Prefer exact match at start
            for i in range(5):
                if content.startswith(str(i)):
                    return i
            # fallback: look for digit
            for i in range(5):
                if str(i) in content:
                    return i
            return None

        for _, row in tqdm(samples, total=10, desc="Classifying"):
            try:
                # --- Sentiment ---
                if data_type == "sentiment":
                    text = str(row[text_column])
                    prompt = f"Classify this text's sentiment as only POSITIVE or NEGATIVE: {text}"
                    def parse_label(content):
                        uc = content.upper()
                        return "POSITIVE" if "POS" in uc else "NEGATIVE"
                    if method == "bert":
                        label, score = classify_with_chunks(text, classifier, tokenizer)
                    else:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }],
                            max_tokens=10
                        )
                        label = parse_label(response.choices[0].message.content.strip())
                        score = 1.0
                    result_data = {
                        "text": text,
                        "label": label,
                        "score": score,
                        "original_data": row.to_dict(),
                        "llm_explanation": "",
                        "shap_plot_explanation": "",
                        "shapwithllm_explanation": "",
                    }
                    if label_column:
                        result_data["actualLabel"] = str(row[label_column]).strip().upper()
                    results.append(result_data)
                    stats["total"] += 1
                    if label == "POSITIVE":
                        stats["positive"] += 1
                    else:
                        stats["negative"] += 1

                # --- Legal / CaseHold style ---
                elif data_type == "legal":
                    context = str(row[text_column])
                    choices = [row.get(f'holding_{i}', '') for i in range(5)]
                    prompt = f"Legal Scenario:\n{context}\n\nSelect the holding that is most supported by the legal scenario above.\n"
                    for idx, holding in enumerate(choices):
                        prompt += f"{idx}: {holding}\n"
                    prompt += "\nReply with the number (0, 1, 2, 3, or 4) only."

                    if method == "bert":
                        label, score = None, 0.0  # or implement MCQ BERT logic
                    else:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }],
                            max_tokens=10
                        )
                        pred = parse_casehold_label(response.choices[0].message.content)
                        label = pred if pred is not None else -1
                        score = 1.0

                    result_data = {
                        "citing_prompt": context,
                        "choices": choices,
                        "label": label,
                        "score": score,
                        "original_data": row.to_dict(),
                        "llm_explanation": "",
                        "shap_plot_explanation": "",
                        "shapwithllm_explanation": "",
                    }
                    if label_column:
                        result_data["actualLabel"] = int(row[label_column])
                    results.append(result_data)
                    stats["total"] += 1
                    if "actualLabel" in result_data and label == result_data["actualLabel"]:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1
                elif data_type == "medical":
                    question = str(row.get("question", ""))
                    context = pretty_pubmed_qa(row.get("context", ""))  # or use 'abstract' if that's your column name
                    long_answer=str(row.get("long_answer", ""))
                    prompt = f"""Given the following context, answer the question as 'yes', or 'no', and reply with just one word.
                
                        Question:
                        {question}
                        
                        Context:
                        {context}
                        
                        Your answer (yes, no) only:
                        """

                    if method == "bert":
                        label, score = None, 0.0  # You can implement a PubMedQA BERT model if you want!
                    else:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "user",
                                "content": prompt
                            }],
                            max_tokens=3
                        )
                        content = response.choices[0].message.content.strip().lower()
                        if "yes" in content:
                            label = "yes"
                        elif "no" in content:
                            label = "no"
                        elif "maybe" in content:
                            label = "maybe"
                        else:
                            label = "maybe"  # fallback
                        score = 1.0

                    result_data = {
                        "question": question,
                        "context": context,
                        "label": label,
                        "long_answer": long_answer,
                        "score": score,
                        "original_data": row.to_dict(),
                        "llm_explanation": "",
                        "shap_plot_explanation": "",
                        "shapwithllm_explanation": "",
                    }
                    if label_column:
                        result_data["actualLabel"] = str(row[label_column]).strip().lower()
                    results.append(result_data)
                    stats["total"] += 1
                    if "actualLabel" in result_data and label == result_data["actualLabel"]:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1

                else:
                    return jsonify({"error": "Invalid data_type"}), 400

            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue

        # --- Metrics ---
        try:
            y_true = []
            y_pred = []
            for r in results:
                pred = str(r.get('label', '')).strip().lower()
                gold = None
                if 'actualLabel' in r:
                    gold = str(r['actualLabel']).strip().lower()
                elif 'original_data' in r:
                    if 'final_decision' in r['original_data']:
                        gold = str(r['original_data']['final_decision']).strip().lower()
                    elif 'label' in r['original_data']:
                        gold = str(r['original_data']['label']).strip().lower()
                if gold is not None and pred != '':
                    y_true.append(gold)
                    y_pred.append(pred)

            # Convert to binary if needed
            if data_type == "sentiment":
                y_true_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_true]
                y_pred_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_pred]
            else:  # medical/PubMedQA or others
                y_true_bin = [1 if x == 'yes' else 0 for x in y_true]
                y_pred_bin = [1 if x == 'yes' else 0 for x in y_pred]

            stats["accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
            stats["precision"] = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
            stats["recall"] = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
            stats["f1_score"] = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
            stats["confusion_matrix"] = {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }

            # Optionally update each result row with actualLabel
            for i, result in enumerate(results):
                if i < len(y_true):
                    result["actualLabel"] = y_true[i]

            # Count predictions for stats
            if data_type == "medical":
                stats["yes"] = sum(1 for x in y_pred if x == "yes")
                stats["no"] = sum(1 for x in y_pred if x == "no")
                stats["maybe"] = sum(1 for x in y_pred if x == "maybe")
            elif data_type == "sentiment":
                stats["positive"] = sum(1 for x in y_pred if x == "positive")
                stats["negative"] = sum(1 for x in y_pred if x == "negative")
            else:
                stats["yes"] = int(sum(y_pred_bin))
                stats["no"] = int(len(y_pred_bin) - sum(y_pred_bin))
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")

        # --- Store results in DB ---
        classification_data = {
            "dataset_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id),
            "method": method,
            "provider": provider if method == 'llm' else None,
            "model": model_name if method == 'llm' else None,
            "data_type": data_type,
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
        traceback.print_exc()
        print(f"Classification error: {str(e)}")

        return jsonify({
            "error": "Classification failed",
            "details": str(e)
        }), 500
def save_explanation_to_db(classification_id, user_id, result_id, explanation_type, content,model_id):
    """Save the explanations generated by the user """
    print(content,model_id,'content and model id ')


    if explanation_type == 'llm':

        update_field = f"results.{result_id}.llm_explanations.{model_id}"
        print('llm saved')


    elif explanation_type == 'shap_plot':
        update_field = f"results.{result_id}.shap_plot_explanation"
        print('shapplot saved')

    elif explanation_type == 'shapwithllm':

        update_field = f"results.{result_id}.shapwithllm_explanations.{model_id}"
        print('shapllm saved')
    else:
        return False



    # Update the classification document
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
    print('and the results',result)

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
            print('llm saved')
        elif explanation_type=='shap_plot':
            field_map = {
                'shap_plot': f'results.{result_id}.shap_plot_explanation',
            }
            print('shapo plot saved')
        else:
            field_map = {
                'shapwithllm': f'results.{result_id}.shapwithllm_explanation'
            }
            print('shap with llm saved')
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
        dataset = load_dataset(hf_dataset_name,  trust_remote_code=True)
        df = dataset["train"].to_pandas()

        # ----------- PATCH: Prettify context column if present -----------
        if "context" in df.columns:
            df["context"] = df["context"].apply(pretty_pubmed_qa)
        # -----------------------------------------------------------------

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
    openrouter_api_key = data.get("openrouter_api", "")


    # Encrypt the user-entered API keys
    encrypted_openai_api = encrypt_api_key(openai_api_key) if openai_api_key else ""
    encrypted_grok_api = encrypt_api_key(grok_api_key) if grok_api_key else ""
    encrypted_deepseek_api = encrypt_api_key(deepseek_api_key) if deepseek_api_key else ""
    encrypted_openrouter_api = encrypt_api_key(openrouter_api_key) if openrouter_api_key else ""

    user_data = {
        'username': data['username'],
        'email': data['email'],
        'password_hash': password_hash,
        'role': 'user',
        'openai_api': encrypted_openai_api,
        'grok_api': encrypted_grok_api,
        'deepseek_api': encrypted_deepseek_api,
        'openrouter_api': encrypted_openrouter_api,
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
    if preferred_model == '':
        print('now it is gpt')
        preferred_model = 'gpt-3.5-turbo'

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
def get_user_api_key_openrouter():
    """Retrieve the user's Groq API key securely."""
    if not current_user.is_authenticated:
        return None

    # Fetch the OpenAI API key from MongoDB to avoid issues with Flask-Login session
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'openrouter_api': 1})
    print(current_user.username,'this is the user')

    if user_data and "openrouter_api" in user_data:
        return decrypt_api_key(user_data['openrouter_api'])  # Return decrypted API key
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
    openrouter_api_key= data.get('openrouter_api')

    update_fields = {}

    # Encrypt and update only if the user provided a new key
    if openai_api_key:
        update_fields["openai_api"] = encrypt_api_key(openai_api_key)

    if grok_api_key:
        update_fields["grok_api"] = encrypt_api_key(grok_api_key)

    if deepseek_api_key:
        update_fields["deepseek_api"] = encrypt_api_key(deepseek_api_key)

    if openrouter_api_key:
        update_fields["openrouter_api"] = encrypt_api_key(openrouter_api_key)

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
            "user_label": " ",
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
        elif provider=='openrouter':
            openai_api_key = get_user_api_key_openrouter()
            client = OpenAI(api_key=openai_api_key)


            if len(text) < 3:
                return jsonify({"error": "Text must be at least 3 characters"}), 400

            # Send the text to OpenAI for sentiment analysis
            prompt = f"Classify the sentiment of the following text as either positive or negative:\n{text}"

            # Call OpenAI GPT-3/4 model to analyze the sentiment
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
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
        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        data = request.json
        print(data)
        prediction_id = data.get('predictionId','fromdata')
        classificationId=data.get('classificationId','empty')
        datatype=data.get('datatype','sentiment')
        if classificationId == 'empty':
            print('empty')
            provider = user_doc.get('preferred_providerex', 'openai')
            model = user_doc.get('preferred_modelex', 'gpt-3.5-turbo')
        else:
            print('not empty')
            provider = data.get('provider', 'openai')
            model = data.get('model', 'gpt-3.5-turbo')
        print(model,provider,'these are model and proviider')

        resultId=data.get('resultId')
        predictedlabel=data.get('predictedlabel')

        truelabel=data.get('truelabel')
        confidence=data.get('confidence')
        text = data.get('text'  )

        explainer_type = data.get('explainer_type', 'llm')

        print('this is predid', prediction_id)

        if prediction_id=='fromdata': #if the explanation is for dataset
            if not text:
                return jsonify({"error": "Missing text"}), 400

            if explainer_type == 'shap':
                explanation_data,top_words = generate_shap_explanation(text, predictedlabel)
                # Save SHAP plot
                save_explanation_to_db(classificationId,current_user.id,resultId,'shap_plot',explanation_data,model)

                return jsonify({'explanation': explanation_data,'explainer_type': explainer_type , 'top_words': top_words})
            else:
                explanation_text = generate_llm_explanationofdataset(text, predictedlabel,truelabel, confidence,provider,model,datatype)
                save_explanation_to_db(classificationId,current_user.id,resultId,'llm',explanation_text,model)

                return jsonify({"explanation": explanation_text,'explainer_type': explainer_type})


        else: # if the explanation is asked through dashboard

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
        print('this is the provider')
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
        elif provider=='groq':
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
        elif provider=='openrouter':
            print('openroutersss')
            openai_api_key = get_user_api_key_openrouter()
            print(openai_api_key,'inst it a key')

            if not openai_api_key:
                return "Error: No OpenAI API key found for this user."

            client = OpenAI(  base_url="https://openrouter.ai/api/v1",
                              api_key=openai_api_key)

            prompt = f"""
                Explain this sentiment analysis result in simple terms:
                
                Text: {text}
                Sentiment: {label} ({score}% confidence)
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            explanation = response.choices[0].message.content
            return explanation


    except Exception as e:
            print(f"Error: {e}")
            return f"Error: {str(e)}"

from flask import request, jsonify
from flask_login import login_required, current_user
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from bson import ObjectId
import traceback
import re

@app.route('/api/classify_and_explain/<dataset_id>', methods=['POST'])
@login_required
def classify_and_explain(dataset_id):
    import re
    from datetime import datetime
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    import pandas as pd

    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    try:
        # --- Extract parameters ---
        method = data.get('method')
        data_type = data.get('dataType', 'sentiment')
        if data_type == 'legal':
            text_column = data.get('text_column', 'citing_prompt')
            label_column = data.get('label_column', 'label')
        elif data_type == 'sentiment':
            text_column = data.get('text_column', 'text')
            label_column = data.get('label_column', 'label')
        else:
            text_column = "question"
            label_column = "final_decision"

        if method not in CLASSIFICATION_METHODS:
            return jsonify({"error": f"Invalid method. Must be one of: {CLASSIFICATION_METHODS}"}), 400

        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        provider = user_doc.get('preferred_provider', 'openai')
        model_name = user_doc.get('preferred_model', 'gpt-3.5-turbo')


        dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # --- Load dataset ---
        try:
            df = pd.read_csv(dataset['filepath'])
            if text_column not in df.columns:
                return jsonify({"error": f"Text column '{text_column}' not found in dataset"}), 400
            if label_column and label_column not in df.columns:
                return jsonify({"error": f"Label column '{label_column}' not found in dataset"}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500

        # --- Init client ---
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
            elif provider == 'ollama':
                print('using ollama')
            else:
                return jsonify({"error": "Invalid LLM provider"}), 400

        # --- Process rows ---
        results = []
        explanations_to_save = []  # To save after we get classification_id
        stats = {"total": 0}
        if data_type == "sentiment":
            stats.update({"positive": 0, "negative": 0})
        elif data_type == "legal":
            stats.update({"correct": 0, "incorrect": 0})
        elif data_type == "medical":
            stats.update({"correct": 0, "incorrect": 0, "maybe": 0})

        sample_size = 15
        print(df[label_column].value_counts(),'dist of labels')
        if label_column in df.columns:
            df_sampled, _ = train_test_split(
                df,
                train_size=sample_size,
                stratify=df[label_column],
                random_state=42,
            )
        else:
            df_sampled = df.sample(n=sample_size, random_state=42)

        samples = list(df_sampled.iterrows())

        for df_idx, row in samples:
            try:
                # --- Prompt Templates ---
                if data_type == "sentiment":
                    text = str(row[text_column])
                    prompt = f"""Given the text below, classify the sentiment as either POSITIVE or NEGATIVE, and briefly explain your reasoning in 2-3 sentences.

                    Text: {text}
                    
                    Format your answer as:
                    Sentiment: <POSITIVE/NEGATIVE>
                    Explanation: <your explanation here>
                    """
                elif data_type == "legal":
                    context = str(row[text_column])
                    choices = [row.get(f'holding_{i}', '') for i in range(5)]
                    holdings_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
                    prompt = f"""Assume you are a legal advisor

                    Statement: {context}
                    Holdings:
                    {holdings_str}
                    select the most appropriate holding (choose 0, 1, 2, 3, or 4)  and explain your recommendation
                    Format your answer as:
                    Holding: <number>
                    Explanation: <your explanation here>
                    """
                elif data_type == "medical":
                    question = str(row.get("question", ""))
                    context = pretty_pubmed_qa(row.get("context", ""))
                    long_answer= str(row.get("long_answer", ""))
                    prompt = f"""Assume you are a Medical advisor 

                    Question: {question}
                    Context: {context}
                    
                    Answer the questions with Yes/No and give an explanation for your recommendation.
                    
                    Format your answer as:
                    Answer: <yes/no/maybe>
                    Explanation: <your explanation here>
                    """
                else:
                    continue  # skip unknown type

                # --- LLM call (classify + explain) ---
                if method == "llm":
                    if provider == "ollama":
                        llm = Ollama(model=model_name)
                        content = llm.invoke([prompt])
                        print(content, 'this is what ollama prompt')
                        # Parse output
                        if data_type == "sentiment":
                            m = re.search(r"Sentiment:\s*(POSITIVE|NEGATIVE)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = m.group(1).upper()
                                explanation = m.group(2).strip()
                            else:
                                lines = content.split('\n')
                                label = lines[0].replace("Sentiment:", "").strip().upper()
                                explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()
                            score = 1.0
                        elif data_type == "legal":
                            m = re.search(r"Holding:\s*([0-4])[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = int(m.group(1))
                                explanation = m.group(2).strip()
                            else:
                                num_match = re.search(r"[0-4]", content)
                                label = int(num_match.group(0)) if num_match else -1
                                explanation = content
                            score = 1.0
                        elif data_type == "medical":
                            m = re.search(r"Answer:\s*(yes|no|maybe)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = m.group(1).lower()
                                explanation = m.group(2).strip()
                            else:
                                lines = content.split('\n')
                                label = lines[0].replace("Answer:", "").strip().lower()
                                explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()
                            score = 1.0
                        else:
                            continue
                        explanations_to_save.append({
                            "df_index": int(df_idx),
                            "explanation": explanation,
                            "model": model_name,
                            "type": "llm",
                        })
                        # Skip further LLM calls for this row
                        # Continue to result handling logic
                        # Use this indentation to match surrounding block
                        # Rest of code continues unchanged
                        # Add an "else" after this for non-Ollama providers (keep existing OpenAI/Groq/etc.)
                        # Example: else: (existing code)
                        # Do not indent the next 'else' branch for OpenAI/Groq
                    else:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0
                        )
                        content = response.choices[0].message.content.strip()
                        print(content, 'this is what llm prompt')

                        # Parse output
                        if data_type == "sentiment":
                            m = re.search(r"Sentiment:\s*(POSITIVE|NEGATIVE)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = m.group(1).upper()
                                explanation = m.group(2).strip()
                            else:
                                lines = content.split('\n')
                                label = lines[0].replace("Sentiment:", "").strip().upper()
                                explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()
                            score = 1.0

                        elif data_type == "legal":
                            m = re.search(r"Holding:\s*([0-4])[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = int(m.group(1))
                                explanation = m.group(2).strip()
                            else:
                                num_match = re.search(r"[0-4]", content)
                                label = int(num_match.group(0)) if num_match else -1
                                explanation = content
                            score = 1.0

                        elif data_type == "medical":
                            m = re.search(r"Answer:\s*(yes|no|maybe)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = m.group(1).lower()
                                explanation = m.group(2).strip()
                            else:
                                lines = content.split('\n')
                                label = lines[0].replace("Answer:", "").strip().lower()
                                explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()
                            score = 1.0
                        else:
                            continue

                        # Save explanation for later, track by DataFrame index
                        explanations_to_save.append({
                            "df_index": int(df_idx),
                            "explanation": explanation,
                            "model": model_name,
                            "type": "llm",
                        })
                else:
                    label, score, explanation = None, 0.0, ""
                if data_type == "medical":

                    result_data = {
                        "label": label,
                        "score": score,
                        "llm_explanation": explanation,
                        "original_data": row.to_dict(),
                    }
                    if data_type == "sentiment":
                        result_data["text"] = text
                    elif data_type == "legal":
                        result_data["citing_prompt"] = context
                        result_data["choices"] = choices
                    elif data_type == "medical":
                        result_data["question"] = question
                        result_data["context"] = context
                        result_data["long_answer"] = long_answer
                        ner_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
                        groq = get_user_api_key_groq()  # Or however your code names these
                        row_reference = {
                            "ground_question": question,
                            "ground_explanation": long_answer,
                            "ground_label": row[label_column],
                            "predicted_explanation": explanation,
                            "predicted_label": label,
                            "ground_context": context,
                        }
                        if provider == "openrouter":
                            api = get_user_api_key_openrouter()
                        elif provider == "openai":
                            api = get_user_api_key_openai()
                        elif provider == "groq":
                            api = get_user_api_key_groq()
                        else:
                            api = 'api'
                        print(provider,'this is the provider')
                        score = lext(
                            context, question, long_answer, row[label_column],
                            model_name, groq, provider, api, ner_pipe, row_reference
                        )
                        # --- Assemble result row ---
                        plausibility_metrics = {
                            "iterative_stability": row_reference.get("iterative_stability"),
                            "paraphrase_stability": row_reference.get("paraphrase_stability"),
                            "consistency": row_reference.get("consistency"),
                            "plausibility": row_reference.get("plausibility")
                        }

                        # Extract Faithfulness Metrics
                        faithfulness_metrics = {
                            "qag_score": row_reference.get("qag_score"),
                            "counterfactual": row_reference.get("counterfactual_scaled"),  # or just "counterfactual"
                            "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                            "faithfulness": row_reference.get("faithfulness")
                        }

                        # Trustworthiness
                        trustworthiness_score = row_reference.get("trustworthiness")
                        metrics = {"plausibility_metrics": plausibility_metrics, "faithfulness_metrics": faithfulness_metrics, "trustworthiness_score": trustworthiness_score}
                        result_data["metrics"] = metrics
                # Include ground truth if present
                if label_column:
                    result_data["actualLabel"] = row[label_column]

                # Add DataFrame index for robust tracking
                result_data["df_index"] = int(df_idx)

                results.append(result_data)
                stats["total"] += 1
                if data_type == "sentiment":
                    if label == "POSITIVE":
                        stats["positive"] += 1
                    else:
                        stats["negative"] += 1
                elif data_type == "legal":
                    if "actualLabel" in result_data and label == result_data["actualLabel"]:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1
                elif data_type == "medical":
                    if label == "maybe":
                        stats["maybe"] += 1
                    if "actualLabel" in result_data and label == result_data["actualLabel"]:
                        stats["correct"] += 1
                    else:
                        stats["incorrect"] += 1

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing row: {str(e)}")
                continue
        model_name = model_name.replace('.', '_') if model_name else None


        # --- Metrics ---
        try:
            y_true = []
            y_pred = []
            for r in results:
                pred = str(r.get('label', '')).strip().lower()
                gold = None
                if 'actualLabel' in r:
                    gold = str(r['actualLabel']).strip().lower()
                elif 'original_data' in r:
                    if 'final_decision' in r['original_data']:
                        gold = str(r['original_data']['final_decision']).strip().lower()
                    elif 'label' in r['original_data']:
                        gold = str(r['original_data']['label']).strip().lower()
                if gold is not None and pred != '':
                    y_true.append(gold)
                    y_pred.append(pred)

            # Convert to binary if needed
            if data_type == "sentiment":
                y_true_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_true]
                y_pred_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_pred]
            else:  # medical/PubMedQA or others
                y_true_bin = [1 if x == 'yes' else 0 for x in y_true]
                y_pred_bin = [1 if x == 'yes' else 0 for x in y_pred]

            stats["accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
            stats["precision"] = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
            stats["recall"] = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
            stats["f1_score"] = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
            stats["confusion_matrix"] = {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }

            # Optionally update each result row with actualLabel
            for i, result in enumerate(results):
                if i < len(y_true):
                    result["actualLabel"] = y_true[i]

            # Count "yes"/"no" for stats
            if data_type == "medical":
                stats["yes"] = sum(1 for x in y_pred if x == "yes")
                stats["no"] = sum(1 for x in y_pred if x == "no")
                stats["maybe"] = sum(1 for x in y_pred if x == "maybe")
            else:
                stats["yes"] = int(sum(y_pred_bin))
                stats["no"] = int(len(y_pred_bin) - sum(y_pred_bin))
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")

        # --- Store all results in DB ---
        classification_data = {
            "dataset_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id),
            "method": method,
            "provider": provider if method == 'llm' else None,
            "model": model_name if method == 'llm' else None,
            "data_type": data_type,
            "results": results,
            "created_at": datetime.now(),
            "stats": stats
        }
        classification_id = mongo.db.classifications.insert_one(classification_data).inserted_id
        explanation_models = [{'provider': provider, 'model': model_name}]
        mongo.db.classifications.update_one(
            {
                "_id": ObjectId(classification_id),
                "user_id": ObjectId(current_user.id)  # Ensure user owns this classification
            },
            {
                "$set": {
                    "explanation_models": explanation_models,
                    "updated_at": datetime.now()
                }
            }
        )

        # --- Save all explanations AFTER classification_id is known ---
        for explanation_entry in explanations_to_save:
            # Find the corresponding result index by matching df_index
            result_id = next(
                (i for i, r in enumerate(results) if r.get("df_index") == explanation_entry["df_index"]),
                None
            )
            if result_id is not None:
                save_explanation_to_db(
                    classification_id=str(classification_id),
                    user_id=current_user.id,
                    result_id=result_id,
                    explanation_type=explanation_entry["type"],
                    content=explanation_entry["explanation"],
                    model_id=explanation_entry["model"].replace('.', '_')
                )

        return jsonify({
            "message": "Classification+explanation completed",
            "classification_id": str(classification_id),
            "stats": stats,
            "sample_count": len(results)
        }), 200

    except Exception as e:
        traceback.print_exc()
        print(f"Classification+explanation error: {str(e)}")
        return jsonify({
            "error": "Classification+explanation failed",
            "details": str(e)
        }), 500
def generate_llm_explanationofdataset(text, label,truelabel, score,provider,model,datatype):
    """Generete generative AI explanation of singel instances in the dataset"""


    try:
        myprompt=''
        if datatype == 'legal':
            myprompt=f"""
                Explain why this holding is correct for the legal statement :
                
                Statement: {text}
                holding: {label} 
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """
        elif datatype == 'sentiment':
             myprompt=f"""
                Explain this sentiment analysis result in simple terms:
                
                Text: {text}
                Sentiment: {label} ({score}% confidence)
                
                Focus on key words and overall tone.
                Keep explanation under 3 sentences.
            """
        else:
            myprompt=f"""
               Given the following biomedical question and its context, explain in simple terms why the answer is {label} from given {score}

                Question:{text}
                
                Context: {score}
                
                Predicted Answer: {label} 
                
            """
        if provider == 'openai':

            openai_api_key = get_user_api_key_openai()

            if not openai_api_key:
                return "Error: No OpenAI API key found for this user."

            client = OpenAI(api_key=openai_api_key)

            prompt = myprompt

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            explanation = response.choices[0].message.content
            return explanation
        elif provider=='openrouter':
            openai_api_key = get_user_api_key_openrouter()

            if not openai_api_key:
                return "Error: No OpenAI API key found for this user."

            client = OpenAI(  base_url="https://openrouter.ai/api/v1",
                              api_key=openai_api_key)


            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": myprompt}]
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
                        "content":myprompt,
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
        print(output_str)

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
        classificationId=data.get('classificationId','empty')
        resultId=data.get('resultId')
        text = data.get('text')
        prediction_id = data.get('predictionId')
        user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
        if classificationId == 'empty':
            provider = user_doc.get('preferred_providerex', 'openai')
            model = user_doc.get('preferred_modelex', 'gpt-3.5-turbo')
        else:
            provider= data.get('provider', 'openai')
            model = data.get('model', 'gpt-3.5-turbo')
        explainer_type = data.get('explainer_type', 'llm')
        if prediction_id:
            print('prediction is in shap')

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

                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',explanation,model)

                return explanation
            elif provider=='grok':
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

                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',chat_completion.choices[0].message.content,model)


                return chat_completion.choices[0].message.content

            elif provider=='openrouter':
                openai_api_key = get_user_api_key_openrouter()

                if not openai_api_key:
                    return "Error: No OpenAI API key found for this user."

                client = OpenAI( base_url="https://openrouter.ai/api/v1",api_key=openai_api_key)

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
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

                explanation = response.choices[0].message.content

                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',explanation,model)

                return explanation


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
                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',explanation,model)

                return explanation
            elif provider=='grok':
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


                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',chat_completion.choices[0].message.content,model)


                return chat_completion.choices[0].message.content
            if provider == 'openrouter':

                openai_api_key = get_user_api_key_openrouter()

                if not openai_api_key:
                    return "Error: No OpenAI API key found for this user."

                client = OpenAI(base_url="https://openrouter.ai/api/v1",api_key=openai_api_key)

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
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

                explanation = response.choices[0].message.content

                save_explanation_to_db(classificationId,current_user.id,resultId,'shapwithllm',explanation,model)

                return explanation

        return jsonify({"error": "Invalid provider or missing return path"}), 400
    except Exception as e:
        traceback.print_exc()  # <-- HATA BURADA GÖRÜNÜR

        print(f"Error: {e}")
        return f"Error: {str(e)}"
@app.route('/api/classificationentry/<classification_id>/<result_id>', methods=['GET'])
@login_required
def get_classificationentry(classification_id, result_id):
    """Get a single entry of a classified dataset, supporting sentiment, legal, and medical."""
    try:
        classification = mongo.db.classifications.find_one({
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(current_user.id)
        })

        if not classification:
            return jsonify({"error": "Classification not found"}), 404

        result = classification['results'][int(result_id)]
        data_type = classification.get('data_type')

        # Universal main text field
        entry_text = result.get('text') or result.get('citing_prompt') or result.get('question') or ''

        # For legal, get holdings
        holdings = []
        if data_type == "legal":
            for i in range(5):
                holding_key = f'holding_{i}'
                if holding_key in result.get('original_data', {}):
                    holdings.append(result['original_data'][holding_key])
        else:
            holdings = None

        # For medical, get question, context, long_answer if present
        question = result.get('question') if data_type == "medical" else None
        context = result.get('context') if data_type == "medical" else None
        long_answer = result.get('long_answer') if data_type == "medical" else None
        trustworthiness_score = result.get("metrics", {}).get("trustworthiness_score") if data_type == "medical" else None

        response_data = {
            "text": entry_text,
            "prediction": result['label'],
            "confidence": result['score'],
            "actualLabel": result.get('actualLabel'),
            "llm_explanation": result.get('llm_explanation', ''),
            "shap_plot": result.get('shap_plot_explanation', ''),
            "shapwithllm": result.get('shapwithllm_explanation', ''),
            "ratings": result.get('ratings', {}),
            "rating_timestamp": result.get('rating_timestamp', ''),
            "provider": classification.get('provider'),
            "model": classification.get('model'),
            "llm_explanations": result.get("llm_explanations", {}),
            "shap_plot_explanation": result.get("shap_plot_explanation"),
            "shapwithllm_explanations": result.get("shapwithllm_explanations", {}),
            "holdings": holdings,
            "data_type": data_type,
            "method": classification.get('method'),
            # For medical
            "question": question,
            "context": context,
            "long_answer": long_answer,
            "trustworthiness_score": trustworthiness_score,
        }
        print(response_data)

        return jsonify(response_data)

    except IndexError:
        return jsonify({"error": "Result not found"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
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
            "timestamp": prediction["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "user_label":prediction["user_label"]
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
def save_ratings_to_db(classification_id, user_id, result_id, ratings, timestamp):
    update_fields = {
        f"results.{result_id}.ratings": ratings,
    }

    result = mongo.db.classifications.update_one(
        {
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(user_id),
            f"results.{result_id}": {"$exists": True}
        },
        {
            "$set": update_fields
        }
    )

    return result.modified_count > 0
@app.route('/api/save_ratings', methods=['POST'])
def save_ratings():
    try:
        data = request.get_json()

        classification_id = data.get("classificationId")
        result_id = data.get("resultId")
        ratings = data.get("ratings")
        timestamp = data.get("timestamp")
        user_id = current_user.id
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)  # depends on your auth system

        if not all([classification_id, result_id, ratings, timestamp, user_id]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400

        success = save_ratings_to_db(classification_id, user_id, result_id, ratings, timestamp)

        if success:
            return jsonify({"success": True}), 200
        else:
            return jsonify({"success": False, "message": "Update failed"}), 404

    except Exception as e:
        print("Error saving ratings:", e)
        return jsonify({"success": False, "message": "Server error"}), 500
def pretty_pubmed_qa(data):
    # If data is not a dict, convert it from Python repr to dict first (may require ast.literal_eval)
    if isinstance(data, str):
        import ast
        try:
            data = ast.literal_eval(data)
        except Exception:
            return data  # fallback: just return the string

    context = "\n".join(data.get('contexts', []))
    labels = ", ".join(data.get('labels', []))
    meshes = ", ".join(data.get('meshes', []))
    rr_pred = ", ".join(data.get('reasoning_required_pred', []))
    rf_pred = ", ".join(data.get('reasoning_free_pred', []))
    return (
        f"Context:\n{context}\n"
        f"Labels: {labels}\n"
        f"MeSH Terms: {meshes}\n"
        f"Reasoning Required Prediction: {rr_pred}\n"
        f"Reasoning Free Prediction: {rf_pred}"
    )
def extract_context_explanation(context_pretty_str):
    # Use regex to capture everything between "Context:" and "Labels:"
    match = re.search(r'Context:\n(.*?)\nLabels:', context_pretty_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

if __name__ == '__main__':
    app.run(port=5000, debug=True)