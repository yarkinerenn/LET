# routes/classify.py
import traceback

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


# ---------- /api/classify_and_explain/<dataset_id> ----------
@classify_bp.route('/api/classify_and_explain/<dataset_id>', methods=['POST'])
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
        limit = data.get('limit')
        if data_type == 'legal':
            text_column = data.get('text_column', 'citing_prompt')
            label_column = data.get('label_column', 'label')
        elif data_type == 'sentiment':
            text_column = data.get('text_column', 'text')
            label_column = data.get('label_column', 'label')
        elif data_type == 'ecqa':
            text_column = "q_text"
            label_column = "q_ans"
        elif data_type == 'snarks':
            text_column = "input"
            label_column = "target"
        elif data_type == 'hotel':
            text_column = "text"
            label_column = "deceptive"
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
                myapi= get_user_api_key_openai()
                if not myapi:
                    return jsonify({"error": "OpenAI API key not configured"}), 400
                client = OpenAI(api_key=myapi)
                print('using ollama')
            elif provider =='openrouter':
                api_key = get_user_api_key_openrouter()
                if not api_key:
                    return jsonify({"error": "Openrouter API key not configured"}), 400
                client = OpenAI( base_url="https://openrouter.ai/api/v1",api_key=api_key)
            elif provider =='gemini':
                api_key = get_user_api_key_gemini()
                print('using gemini', api_key)
                if not api_key:
                    return jsonify({"error": "Gemini API key not configured"}), 400
                client = OpenAI( base_url="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=api_key)
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
        elif data_type == "hotel":
            stats.update({"correct": 0, "incorrect": 0})
        elif data_type == "ecqa":
            stats.update({})  # multiclass; will update after

        sample_size = limit
        print(df[label_column].value_counts(),'dist of labels')
        if data_type == "ecqa" or data_type == "snarks":
            # For ECQA, don't stratify, just sample N entries
            df_sampled = df.sample(n=sample_size, random_state=42)
        elif data_type == "hotel":
            # stratify by BOTH gold label and polarity
            strat_cols = [label_column, "polarity"]
            if all(c in df.columns for c in strat_cols):
                strata = df[strat_cols].astype(str).agg("||".join, axis=1)
                try:
                    df_sampled, _ = train_test_split(
                        df,
                        train_size=sample_size,
                        stratify=strata,
                        random_state=42,
                    )
                except ValueError:
                    # Fallback if some strata have too few samples
                    df_sampled = df.sample(n=sample_size, random_state=42)
            else:
                df_sampled = df.sample(n=sample_size, random_state=42)
        elif label_column in df.columns:
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
                    if provider!='ollama':
                        prompt = f"""Assume you are a Medical advisor 
    
                        Question: {question}
                        Context: {context}
                        
                        Answer the questions with Yes/No/maybe and give an explanation for your recommendation.
                        
                        Format your answer as:
                        Answer: <yes/no/maybe>
                        Explanation: <your explanation here>
                        """

                    else:
                        prompt=f"""Assume you are a Medical advisor 
    
                        Question: {question}
                        Context: {context}
                        
                        Answer the questions with Yes/No/maybe and give an explanation for your recommendation.
                        """
                elif data_type == "ecqa":
                    question = str(row[text_column])
                    choices = [row.get('q_op1', ''), row.get('q_op2', ''), row.get('q_op3', ''), row.get('q_op4', ''), row.get('q_op5', '')]
                    gold_label = row[label_column]
                    long_answer = row.get("taskB", "")
                    prompt = f"""Given the following question and five answer options, select the best answer and explain your choice in 2-3 sentences. YOU MUST ONLY CHOOSE ONE OF THE CHOICES

                    Question: {question}
                    
                    Choices:
                    1. {choices[0]}
                    2. {choices[1]}
                    3. {choices[2]}
                    4. {choices[3]}
                    5. {choices[4]}
                    
                    Format your answer as:
                    Answer: <Choice as number >
                    Explanation: <your explanation here>
                    """
                elif data_type == "snarks":
                    question = str(row[text_column])
                    gold_label = row[label_column]
                    prompt = f"""You are a sarcasm detection system. You will chose (A) or (B) as your answer and explain your decision in 2-3 sentences. Do not quote from the question or mention any words in your explanation.
    
                    Question: {question}
                    
                    Format your answer as:
                    Answer: <Choice as (A) or (B)>
                    Explanation: <your explanation here>
                    
                    An Example: 
                    Which statement is sarcastic?
                    Options:
                    (A) yeah just jump from the mountain like everybody else.
                    (B) yeah just jump from the mountain like everybody else you have a parachute too.
                    
                    Answer: <(A)>
                    Explanation: <The statement is sarcastic because it is criticizes one should not do what everybody does but think first>
                    """
                elif data_type == "hotel":
                    question = str(row[text_column])
                    gold_label = row[label_column]
                    prompt = f"""You are a deceptive hotel review detection system. You will chose "truthful" or "deceptive" as your answer and explain your decision in 2-3 sentences.
    
                    Question: {question}
                    
                    Format your answer as:
                    Answer: <Choice as "truthful" or "deceptive">
                    Explanation: <your explanation here>
                    """
                else:
                    continue  # skip unknown type

                # --- LLM call (classify + explain) ---
                if method == "llm":
                    if provider == "ollama":
                        llm = Ollama(model=model_name,temperature=0)
                        content = llm.invoke(prompt)
                        # print(content, 'this is what ollama prompt')
                        #
                        # gatherer_prompt=f''' Without changing anything, format this '{content}' answer as:
                        # Answer: <whatever the answer is>
                        # Explanation: <explanation here>'''
                        #
                        # response = client.chat.completions.create(
                        #     model='gpt-5-nano-2025-08-07',
                        #     messages=[{"role": "user", "content":gatherer_prompt }],
                        #     temperature=0
                        # )
                        # content = response.choices[0].message.content.strip()
                        print(content, 'this is what ollama prompt after gpt')
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
                        elif data_type == "ecqa" or data_type == "snarks" or data_type == "hotel":
                            # Example expected format:
                            # Answer: <answer text>
                            # Explanation: <explanation here>
                            m = re.search(r"Answer:\s*(.+)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = m.group(1).strip()
                                explanation = m.group(2).strip()
                            else:
                                lines = content.split('\n')
                                label = lines[0].replace("Answer:", "").strip()
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
                            messages=[{"role": "user", "content": prompt}]                        )
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
                        elif data_type == "ecqa" or data_type == "snarks" or data_type == "hotel":
                            # Example expected format:
                            # Answer: <answer text>
                            # Explanation: <explanation here>
                            m = re.search(r"Answer:\s*(.+)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
                            if m:
                                label = m.group(1).strip()
                                explanation = m.group(2).strip()
                            else:
                                lines = content.split('\n')
                                label = lines[0].replace("Answer:", "").strip()
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
                # --- Assemble result_data for medical and ecqa types ---
                if provider == "openrouter":
                    api = get_user_api_key_openrouter()
                elif provider == "openai":
                    api = get_user_api_key_openai()
                elif provider == "groq":
                    api = get_user_api_key_groq()
                elif provider == "gemini":
                    api = get_user_api_key_gemini()
                else:
                    api = 'api'
                open_ai_api=get_user_api_key_openai()
                groq = get_user_api_key_groq()
                if data_type == "medical":
                    result_data = {
                        "question": question,
                        "context": context,
                        "long_answer": long_answer,
                        "label": label,
                        "score": score,
                        "llm_explanation": explanation,
                        "actualLabel": row[label_column],
                        "df_index": int(df_idx),
                    }
                    # Compute trustworthiness & metrics here
                    ner_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
                    row_reference = {
                        "ground_question": question,
                        "ground_explanation": long_answer,
                        "ground_label": row[label_column],
                        "predicted_explanation": explanation,
                        "predicted_label": label,
                        "ground_context": context,
                    }

                    trust_score = lext(
                        context, question, long_answer, row[label_column],
                        model_name, groq, provider, api, ner_pipe,data_type, row_reference
                    )
                    print(row_reference)
                    plausibility_metrics = {
                        "iterative_stability": row_reference.get("iterative_stability"),
                        "paraphrase_stability": row_reference.get("paraphrase_stability"),
                        "consistency": row_reference.get("consistency"),
                        "plausibility": row_reference.get("plausibility")
                    }
                    faithfulness_metrics = {
                        "qag_score": row_reference.get("qag_score"),
                        "counterfactual": row_reference.get("counterfactual_scaled"),
                        "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                        "faithfulness": row_reference.get("faithfulness")
                    }
                    trustworthiness_score = row_reference.get("trustworthiness")
                    metrics = {
                        "plausibility_metrics": plausibility_metrics,
                        "faithfulness_metrics": faithfulness_metrics,
                        "trustworthiness_score": trustworthiness_score,
                        "lext_score": trust_score
                    }
                    result_data["metrics"] = metrics
                elif data_type == "ecqa":
                    question = str(row[text_column])
                    choices = [row.get('q_op1', ''), row.get('q_op2', ''), row.get('q_op3', ''), row.get('q_op4', ''), row.get('q_op5', '')]
                    gold_label = row[label_column]
                    ground_explanation = row.get("taskB", "")
                    result_data = {
                        "question": question,
                        "choices": choices,
                        "label": label,
                        "score": score,
                        "llm_explanation": explanation,
                        "actualLabel": gold_label,
                        "ground_explanation": ground_explanation,
                        "df_index": int(df_idx),
                    }
                    row_reference = {
                        "ground_question": question,
                        "ground_explanation": ground_explanation,
                        "ground_label": gold_label,
                        "predicted_explanation": explanation,
                        "predicted_label": label,
                    }
                    context = None

                    # --- DEBUG: Log inputs to LExT for ECQA ---
                    def _mask_key(k):
                        try:
                            if not k:
                                return str(k)
                            s = str(k)
                            return s[:4] + '...' + s[-2:] if len(s) > 8 else '***'
                        except Exception:
                            return '<unprintable>'

                    print('[LExT DEBUG][ECQA] context_len         =', 0 if context is None else len(str(context)))
                    print('[LExT DEBUG][ECQA] question_preview    =', (question[:120] + '...') if isinstance(question, str) and len(question) > 120 else question)
                    print('[LExT DEBUG][ECQA] ground_expl    =', 0 if not ground_explanation else str(ground_explanation))
                    print('[LExT DEBUG][ECQA] gold_label          =', gold_label)
                    print('[LExT DEBUG][ECQA] model_name          =', model_name)
                    print('[LExT DEBUG][ECQA] provider            =', provider)
                    print('[LExT DEBUG][ECQA] groq_key            =', _mask_key(groq))
                    print('[LExT DEBUG][ECQA] api_key             =', _mask_key(api))
                    print('[LExT DEBUG][ECQA] ner_pipe            =', None)
                    print('[LExT DEBUG][ECQA] row_reference_keys  =', list(row_reference.keys()))

                    trust_score = lext(
                        context, question, ground_explanation, gold_label,
                        model_name, groq, provider, api, None,data_type, row_reference
                    )
                    print(row_reference)

                    plausibility_metrics = {
                        "correctness": row_reference.get("correctness"),
                        "consistency": row_reference.get("consistency"),
                        "plausibility": row_reference.get("plausibility")
                    }
                    faithfulness_metrics = {
                        "qag_score": row_reference.get("qag_score"),
                        "counterfactual": row_reference.get("counterfactual_scaled"),
                        "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                        "faithfulness": row_reference.get("faithfulness")
                    }
                    trustworthiness_score = row_reference.get("trustworthiness")
                    metrics = {
                        "plausibility_metrics": plausibility_metrics,
                        "faithfulness_metrics": faithfulness_metrics,
                        "trustworthiness_score": trustworthiness_score,
                        "lext_score": trust_score
                    }
                    result_data["metrics"] = metrics
                elif data_type == "snarks":
                    question = str(row[text_column])
                    gold_label = row[label_column]
                    result_data = {
                        "question": question,
                        "label": label,
                        "score": score,
                        "llm_explanation": explanation,
                        "actualLabel": gold_label,
                        "df_index": int(df_idx),
                    }
                    row_reference = {
                        "ground_question": question,
                        "ground_explanation": None,
                        "ground_label": gold_label,
                        "predicted_explanation": explanation,
                        "predicted_label": label,
                    }
                    context = None

                    faithfulness_score = faithfulness(
                        explanation, label, question, gold_label,None,
                        groq, model_name, provider, api,data_type, row_reference
                    )
                    print(row_reference)

                    faithfulness_metrics = {
                        "qag_score": row_reference.get("qag_score"),
                        "counterfactual": row_reference.get("counterfactual_scaled"),
                        "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                        "faithfulness": faithfulness_score
                    }
                    metrics = {
                        "faithfulness_metrics": faithfulness_metrics,
                    }
                    result_data["metrics"] = metrics
                elif data_type == "hotel":
                    question = str(row[text_column])
                    gold_label = row[label_column]
                    result_data = {
                        "question": question,
                        "label": label,
                        "score": score,
                        "llm_explanation": explanation,
                        "actualLabel": gold_label,
                        "df_index": int(df_idx),
                    }
                    row_reference = {
                        "ground_question": question,
                        "ground_explanation": None,
                        "ground_label": gold_label,
                        "predicted_explanation": explanation,
                        "predicted_label": label,
                    }
                    context = None

                    faithfulness_score = faithfulness(
                        explanation, label, question, gold_label,None,
                        groq, model_name, provider, api,data_type, row_reference
                    )
                    print(row_reference)

                    faithfulness_metrics = {
                        "qag_score": row_reference.get("qag_score"),
                        "counterfactual": row_reference.get("counterfactual_scaled"),
                        "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                        "faithfulness": faithfulness_score
                    }
                    metrics = {
                        "faithfulness_metrics": faithfulness_metrics,
                    }
                    result_data["metrics"] = metrics
                else:
                    # Existing logic for other types
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

            # --- Updated metrics block: 3-class for medical, binary for sentiment, multiclass for ecqa, else binary ---
            if data_type == "sentiment":
                y_true_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_true]
                y_pred_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_pred]
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
            elif data_type == "medical":
                labels = ["yes", "no", "maybe"]
                stats["accuracy"] = accuracy_score(y_true, y_pred)
                stats["precision"] = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                stats["recall"] = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                stats["f1_score"] = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                stats["confusion_matrix"] = {
                    "labels": labels,
                    "matrix": cm.tolist()
                }
            elif data_type == "ecqa":
                # Multiclass: use unique gold labels as class set
                labels = sorted(list(set(y_true + y_pred)))
                stats["accuracy"] = accuracy_score(y_true, y_pred)
                stats["precision"] = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                stats["recall"] = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                stats["f1_score"] = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                stats["confusion_matrix"] = {
                    "labels": labels,
                    "matrix": cm.tolist()
                }
            elif data_type == "snarks":
                # Binary metrics for snarks: answers are (A) or (B). Treat A as positive (1), B as negative (0).
                def _snarks_to_bin(x):
                    if x is None:
                        return None
                    t = str(x).strip().upper()
                    # remove common wrappers
                    t = t.replace('(', '').replace(')', '').replace('.', '').replace(' ', '')
                    # normalize prefixes like CHOICEA / OPTIONA / ANSWERA
                    t = re.sub(r'^(CHOICE|OPTION|ANSWER)', '', t)
                    if t == 'A':
                        return 1
                    if t == 'B':
                        return 0
                    return None

                y_true_bin_raw = [_snarks_to_bin(x) for x in y_true]
                y_pred_bin_raw = [_snarks_to_bin(x) for x in y_pred]

                # keep only pairs where both labels are recognized
                pairs = [(t, p) for t, p in zip(y_true_bin_raw, y_pred_bin_raw) if t is not None and p is not None]
                if len(pairs) == 0:
                    # If nothing recognizable, set zeros and a degenerate confusion matrix
                    stats["accuracy"] = 0.0
                    stats["precision"] = 0.0
                    stats["recall"] = 0.0
                    stats["f1_score"] = 0.0
                    stats["confusion_matrix"] = {
                        "true_negative": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "true_positive": 0
                    }
                else:
                    y_true_bin, y_pred_bin = map(list, zip(*pairs))
                    stats["accuracy"]  = accuracy_score(y_true_bin, y_pred_bin)
                    stats["precision"] = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                    stats["recall"]    = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                    stats["f1_score"]  = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
                    stats["confusion_matrix"] = {
                        "true_negative": int(tn),
                        "false_positive": int(fp),
                        "false_negative": int(fn),
                        "true_positive": int(tp)
                    }
            elif data_type == "hotel":
                y_true_bin, y_pred_bin = [], []
                # Binary metrics for hotel deception detection: "truthful" vs "deceptive".
                # Map truthful -> 1, deceptive -> 0
                def _hotel_to_bin(x):
                    if x is None:
                        return None
                    t = str(x).strip().lower()
                    # remove punctuation and extra spaces
                    t = re.sub(r"[^a-z]", "", t)
                    if t == "truthful":
                        return 1
                    if t == "deceptive":
                        return 0
                    return None

                y_true_bin_raw = [_hotel_to_bin(x) for x in y_true]
                y_pred_bin_raw = [_hotel_to_bin(x) for x in y_pred]

                # Keep only valid pairs
                pairs = [(t, p) for t, p in zip(y_true_bin_raw, y_pred_bin_raw) if t is not None and p is not None]
                if len(pairs) == 0:
                    # No valid pairs -> define zeros and a degenerate confusion matrix to avoid ravel() crash
                    stats["accuracy"] = 0.0
                    stats["precision"] = 0.0
                    stats["recall"] = 0.0
                    stats["f1_score"] = 0.0
                    stats["confusion_matrix"] = {
                        "true_negative": 0,
                        "false_positive": 0,
                        "false_negative": 0,
                        "true_positive": 0
                    }
                else:
                    y_true_bin, y_pred_bin = map(list, zip(*pairs))
                    stats["accuracy"]  = accuracy_score(y_true_bin, y_pred_bin)
                    stats["precision"] = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                    stats["recall"]    = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                    stats["f1_score"]  = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
                    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
                    stats["confusion_matrix"] = {
                        "true_negative": int(tn),
                        "false_positive": int(fp),
                        "false_negative": int(fn),
                        "true_positive": int(tp)
                    }
            else:
                # For legal and other types, keep previous logic if needed
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
            elif data_type == "ecqa":
                # Count each label
                from collections import Counter
                pred_counts = Counter(y_pred)
                for label in set(y_pred):
                    stats[label] = pred_counts[label]
            elif data_type == "snarks":
                stats["(A)"] = int(sum(y_pred_bin))
                stats["(B)"] = int(len(y_pred_bin) - sum(y_pred_bin))
            else:
                stats["(A)"] = int(sum(y_pred_bin))
                stats["(B)"] = int(len(y_pred_bin) - sum(y_pred_bin))
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