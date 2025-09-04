# routes/classify.py
import traceback
import re
import pandas as pd
from datasets import tqdm
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from bson import ObjectId
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import pipeline,AutoModelForSequenceClassification,AutoTokenizer

from groq import Groq
from openai import OpenAI
from sklearn.model_selection import train_test_split
from langchain_community.llms import Ollama
from extensions import mongo
from LExT.metrics.faithfulness import faithfulness
from LExT.metrics.trustworthiness import lext

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
from .auth import (
    get_user_api_key_openai,
    get_user_api_key_openrouter,
    get_user_api_key_groq, get_user_api_key_deepseek_api, get_user_api_key_gemini,
)
from .helpers import classify_with_chunks, pretty_pubmed_qa, save_explanation_to_db

classify_bp = Blueprint("classify", __name__)
CLASSIFICATION_METHODS = ['bert', 'llm']

# ---------- /api/classify/<dataset_id> ----------
@classify_bp.route('/api/classify/<dataset_id>', methods=['POST'])
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
        elif data_type == 'snarks':
            text_column='input'
            label_column='target'
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


# ---------- /api/analyze ----------
@classify_bp.route('/api/analyze', methods=['POST'])
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
# ---------- /api/analyzewithllm ----------
@classify_bp.route('/api/analyzewithllm', methods=['POST'])
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


