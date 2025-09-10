# ---------- /api/classify_only/<dataset_id> ----------
import traceback
import re
import pandas as pd
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from bson import ObjectId
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from groq import Groq
from openai import OpenAI
from langchain_community.llms import Ollama
from extensions import mongo
from .auth import (
    get_user_api_key_openai,
    get_user_api_key_openrouter,
    get_user_api_key_groq, 
    get_user_api_key_deepseek_api, 
    get_user_api_key_gemini,
)
from .helpers import pretty_pubmed_qa

classify_only_bp = Blueprint("classify_only", __name__)
CLASSIFICATION_METHODS = ['bert', 'llm']


@classify_only_bp.route('/api/classify_only/<dataset_id>', methods=['POST'])
@login_required
def classify_only(dataset_id):
    try:
        # Extract and validate request data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Process the classification request
        result = process_classification_only_request(dataset_id, data, current_user)
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Classification failed", "details": str(e)}), 500


def process_classification_only_request(dataset_id, data, current_user):
    """Main function to process classification-only requests"""
    # Validate method
    method = data.get('method')
    if method not in CLASSIFICATION_METHODS:
        raise ValueError(f"Invalid method. Must be one of: {CLASSIFICATION_METHODS}")

    # Get data type and column mappings
    data_type = data.get('dataType', 'sentiment')
    cot_enabled = data.get('cot', False)
    text_column, label_column = get_column_mappings(data_type, data)

    # Get user preferences
    user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})
    provider = user_doc.get('preferred_provider', 'openai')
    model_name = user_doc.get('preferred_model', 'gpt-3.5-turbo')

    # Load dataset
    dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})
    if not dataset:
        raise ValueError("Dataset not found")

    df = load_and_validate_dataset(dataset['filepath'], text_column, label_column)

    # Initialize client if using LLM method
    client = initialize_llm_client(method, provider) if method == 'llm' else None

    # Sample data based on data type
    sample_size = data.get('limit')
    df_sampled = sample_dataset(df, data_type, label_column, sample_size)

    # Process each sample (classification only)
    results, stats = process_samples_only(
        df_sampled, method, data_type, text_column, label_column,
        client, provider, model_name, cot_enabled
    )

    # Calculate metrics
    stats = calculate_metrics(results, data_type, stats)

    # Store results in database
    classification_id = store_classification_results(
        dataset_id, current_user.id, method, provider,
        model_name, data_type, results, stats
    )

    return {
        "message": "Classification completed",
        "classification_id": str(classification_id),
        "stats": stats,
        "sample_count": len(results)
    }


def get_column_mappings(data_type, data):
    """Get appropriate column mappings based on data type"""
    if data_type == 'legal':
        return (data.get('text_column', 'citing_prompt'), data.get('label_column', 'label'))
    elif data_type == 'sentiment':
        return (data.get('text_column', 'text'), data.get('label_column', 'label'))
    elif data_type == 'ecqa':
        return ("q_text", "q_ans")
    elif data_type == 'snarks':
        return ("input", "target")
    elif data_type == 'hotel':
        return ("text", "deceptive")
    else:
        return ("question", "final_decision")


def load_and_validate_dataset(filepath, text_column, label_column):
    """Load dataset and validate required columns exist"""
    try:
        df = pd.read_csv(filepath)

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if label_column and label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in dataset")

        return df
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")


def initialize_llm_client(method, provider):
    """Initialize and return a chat-completions compatible client for the given provider."""
    if method != 'llm':
        return None

    prov = (provider or "").lower()

    if prov == 'openai':
        api_key = get_user_api_key_openai()
        if not api_key:
            raise ValueError("OpenAI API key not configured")
        return OpenAI(api_key=api_key)

    elif prov == 'groq':
        api_key = get_user_api_key_groq()
        if not api_key:
            raise ValueError("Groq API key not configured")
        return Groq(api_key=api_key)

    elif prov == 'deepseek':
        api_key = get_user_api_key_deepseek_api()
        if not api_key:
            raise ValueError("Deepseek API key not configured")
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    elif prov == 'ollama':
        return None

    elif prov == 'openrouter':
        api_key = get_user_api_key_openrouter()
        if not api_key:
            raise ValueError("Openrouter API key not configured")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    elif prov == 'gemini':
        api_key = get_user_api_key_gemini()
        if not api_key:
            raise ValueError("Gemini API key not configured")
        return OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key)

    else:
        raise ValueError("Invalid LLM provider")


def sample_dataset(df, data_type, label_column, sample_size):
    """Sample the dataset based on data type requirements"""
    if data_type in ["ecqa", "snarks", "legal"]:
        return df.sample(n=sample_size, random_state=42)

    elif data_type == "hotel":
        strat_cols = [label_column, "polarity"]
        if all(c in df.columns for c in strat_cols):
            strata = df[strat_cols].astype(str).agg("||".join, axis=1)
            try:
                df_sampled, _ = train_test_split(
                    df, train_size=sample_size, stratify=strata, random_state=42
                )
                return df_sampled
            except ValueError:
                return df.sample(n=sample_size, random_state=42)

    elif label_column in df.columns:
        df_sampled, _ = train_test_split(
            df, train_size=sample_size, stratify=df[label_column], random_state=42
        )
        return df_sampled

    return df.sample(n=sample_size, random_state=42)


def process_samples_only(df_sampled, method, data_type, text_column, label_column, client, provider, model_name, cot_enabled=False):
    """Process all samples in the dataset (classification only)"""
    results = []
    stats = initialize_stats(data_type)

    for df_idx, row in df_sampled.iterrows():
        try:
            result = process_single_sample_only(
                row, df_idx, method, data_type, text_column, label_column,
                client, provider, model_name, cot_enabled
            )

            if result:
                results.append(result)
                update_stats(stats, data_type, result)

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing row: {str(e)}")
            continue

    return results, stats


def initialize_stats(data_type):
    """Initialize statistics based on data type"""
    stats = {"total": 0}

    if data_type == "sentiment":
        stats.update({"positive": 0, "negative": 0})
    elif data_type == "legal":
        stats.update({"correct": 0, "incorrect": 0})
    elif data_type == "medical":
        stats.update({"correct": 0, "incorrect": 0, "maybe": 0})
    elif data_type == "ecqa":
        stats.update({"correct": 0, "incorrect": 0})
    elif data_type == "hotel":
        stats.update({"correct": 0, "incorrect": 0})

    return stats


def process_single_sample_only(row, df_idx, method, data_type, text_column, label_column, client, provider, model_name, cot_enabled=False):
    """Process a single sample row (classification only)"""
    # Generate appropriate prompt
    prompt = generate_prompt(data_type, row, text_column, provider, cot_enabled)

    # Get LLM response if using LLM method
    if method == 'llm':
        label, score = get_llm_response_only(
            client, provider, model_name, prompt, data_type
        )
    else:
        label, score = None, 0.0

    # Prepare result data based on data type
    result_data = prepare_result_data_only(
        data_type, row, label_column, label, score, df_idx
    )

    return result_data


def generate_prompt(data_type, row, text_column, provider, cot_enabled=False):
    """Generate appropriate prompt based on data type (simplified for classification only)"""
    
    if data_type == "sentiment":
        text = str(row[text_column])
        return f"""Classify the sentiment of the following text as either POSITIVE or NEGATIVE.

        Text: {text}

        Answer with only: POSITIVE or NEGATIVE
        """

    elif data_type == "legal":
        context = str(row[text_column])
        choices = [row.get(f'holding_{i}', '') for i in range(5)]
        holdings_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        return f"""Select the most appropriate holding for this legal statement.

        Statement: {context}
        Holdings:
        {holdings_str}
        
        Answer with only the number (0, 1, 2, 3, or 4):
        """

    elif data_type == "medical":
        question = str(row.get("question", ""))
        context = pretty_pubmed_qa(row.get("context", ""))
         
        return f"""Answer the medical question with Yes, No, or Maybe based on the context.

        Question: {question}
        Context: {context}

        Answer with only: yes, no, or maybe
        """

    elif data_type == "ecqa":
        question = str(row[text_column])
        choices = [row.get('q_op1', ''), row.get('q_op2', ''), row.get('q_op3', ''), row.get('q_op4', ''), row.get('q_op5', '')]
        
        if cot_enabled:
            return f"""You are solving a commonsense multiple-choice question. 
                Think through the problem step by step, considering why each option may or may not be correct. 
                Then state your final answer.

                Question: {question}

                Choices:
                {choices[0]}
                {choices[1]}
                {choices[2]}
                {choices[3]}
                {choices[4]}

                Think step by step, Answer with your choice exactly as written:
                """
        else:
            return f"""Select the best answer from the choices.

            Question: {question}

            Choices:
             {choices[0]}
             {choices[1]}
             {choices[2]}
             {choices[3]}
             {choices[4]}

            Answer with your choice exactly as written:
            """

    elif data_type == "snarks":
        question = str(row[text_column])
        return f"""Determine which statement is sarcastic.

        Question: {question}

        Answer with: (A) or (B)
        """

    elif data_type == "hotel":
        question = str(row[text_column])
        return f"""Classify this hotel review as truthful or deceptive.

        Review: {question}

        Answer with: truthful or deceptive
        """
            
    return ""


def get_llm_response_only(client, provider, model_name, prompt, data_type):
    """Get response from LLM based on provider (classification only)"""
    if provider == "ollama":
        return get_ollama_response_only(prompt, data_type, model_name)
    else:
        return get_standard_llm_response_only(client, model_name, prompt, data_type)


def get_standard_llm_response_only(client, model_name, prompt, data_type):
    """Get response from standard LLM APIs (classification only)"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    
    return parse_llm_response_only(content, data_type)


def get_ollama_response_only(prompt, data_type, model_name):
    """Get response from Ollama (classification only)"""
    llm = Ollama(model=model_name, temperature=0)
    content = llm.invoke(prompt)
    
    return parse_llm_response_only(content, data_type)


def parse_llm_response_only(content, data_type):
    """Parse LLM response based on data type (classification only)"""
    
    if data_type == "sentiment":
        # Look for POSITIVE or NEGATIVE
        if "POSITIVE" in content.upper():
            return "POSITIVE", 1.0
        elif "NEGATIVE" in content.upper():
            return "NEGATIVE", 1.0
        else:
            return content.strip().upper(), 1.0
    
    elif data_type == "legal":
        # Look for numbers 0-4
        num_match = re.search(r"[0-4]", content)
        if num_match:
            return int(num_match.group(0)), 1.0
        else:
            return -1, 1.0
    
    elif data_type == "medical":
        # Look for yes/no/maybe
        content_lower = content.lower()
        if "yes" in content_lower:
            return "yes", 1.0
        elif "no" in content_lower:
            return "no", 1.0
        elif "maybe" in content_lower:
            return "maybe", 1.0
        else:
            return content.strip().lower(), 1.0
    
    elif data_type == "snarks":
        # Look for (A) or (B)
        if "(A)" in content.upper() or " A " in content.upper():
            return "(A)", 1.0
        elif "(B)" in content.upper() or " B " in content.upper():
            return "(B)", 1.0
        else:
            return content.strip(), 1.0
    
    elif data_type == "hotel":
        # Look for truthful or deceptive
        content_lower = content.lower()
        if "truthful" in content_lower:
            return "truthful", 1.0
        elif "deceptive" in content_lower:
            return "deceptive", 1.0
        else:
            return content.strip().lower(), 1.0
    
    elif data_type == "ecqa":
        # Parse ECQA response - handle both CoT and non-CoT formats
        # CoT format: Explanation: ... Answer: ...
        # Non-CoT format: Answer: ... (or just the answer)
        
        # Try to extract answer from CoT format first
        m = re.search(r"Answer:\s*(.+)", content, re.IGNORECASE | re.DOTALL)
        if m:
            answer = m.group(1).strip()
            # Clean up the answer to get just the choice
            if answer.upper() in ['A', 'B', 'C', 'D', 'E']:
                return answer.upper(), 1.0
            else:
                return answer, 1.0
        else:
            # Fallback: look for A, B, C, D, E in the content
            for choice in ['A', 'B', 'C', 'D', 'E']:
                if choice in content.upper():
                    return choice, 1.0
            return content.strip(), 1.0
    
    else:  # other data types
        return content.strip(), 1.0


def prepare_result_data_only(data_type, row, label_column, label, score, df_idx):
    """Prepare result data based on data type (classification only)"""
    
    # Base result data
    result_data = {
        "label": label,
        "score": score,
        "df_index": int(df_idx)
    }
    
    # Add ground truth if present
    if label_column:
        result_data["actualLabel"] = row[label_column]
    
    # Add minimal data type specific fields
    if data_type == "medical":
        result_data.update({
            "question": str(row.get("question", "")),
            "context": pretty_pubmed_qa(row.get("context", "")),
            "long_answer": str(row.get("long_answer", ""))
        })
    elif data_type == "ecqa":
        result_data.update({
            "question": str(row["q_text"]),
            "choices": [row.get('q_op1', ''), row.get('q_op2', ''), row.get('q_op3', ''), row.get('q_op4', ''), row.get('q_op5', '')],
        })
    elif data_type == "snarks":
        result_data["question"] = str(row["input"])
    elif data_type == "hotel":
        result_data["question"] = str(row["text"])
    elif data_type == "sentiment":
        result_data["text"] = str(row["text"])
    elif data_type == "legal":
        result_data.update({
            "citing_prompt": str(row["citing_prompt"]),
            "holdings": [row.get(f'holding_{i}', '') for i in range(5)]
        })

    return result_data


def update_stats(stats, data_type, result):
    """Update statistics based on result"""
    stats["total"] += 1

    if data_type == "sentiment":
        if result["label"] == "POSITIVE":
            stats["positive"] += 1
        else:
            stats["negative"] += 1

    elif data_type in ["legal", "medical", "ecqa", "hotel"]:
        if "actualLabel" in result and result["label"] == result["actualLabel"]:
            stats["correct"] += 1
        else:
            stats["incorrect"] += 1

        if data_type == "medical" and result["label"] == "maybe":
            stats["maybe"] += 1


def calculate_metrics(results, data_type, stats):
    """Calculate performance metrics"""
    y_true = []
    y_pred = []

    # Extract true and predicted labels
    for r in results:
        pred = str(r.get('label', '')).strip().lower()
        gold = extract_gold_label(r, data_type)

        if gold is not None and pred != '':
            y_true.append(gold)
            y_pred.append(pred)

    # Calculate metrics based on data type
    if data_type == "sentiment":
        stats = calculate_sentiment_metrics(y_true, y_pred, stats)
    elif data_type == "medical":
        stats = calculate_medical_type_metrics(y_true, y_pred, stats)
    elif data_type == "ecqa":
        stats = calculate_ecqa_metrics_type(y_true, y_pred, stats)
    elif data_type == "snarks":
        stats = calculate_snarks_metrics_type(y_true, y_pred, stats)
    elif data_type == "hotel":
        stats = calculate_hotel_metrics_type(y_true, y_pred, stats)
    elif data_type == "legal":
        stats = calculate_legal_metrics_type(y_true, y_pred, stats)
    else:
        stats = calculate_binary_metrics(y_true, y_pred, stats)

    # Count predictions for stats
    stats = count_predictions(y_true, y_pred, data_type, stats)

    # Add actual labels to results
    for i, result in enumerate(results):
        if i < len(y_true):
            result["actualLabel"] = y_true[i]

    return stats


def extract_gold_label(result, data_type):
    """Extract the gold standard label from result"""
    if 'actualLabel' in result:
        return str(result['actualLabel']).strip().lower()
    return None


def calculate_sentiment_metrics(y_true, y_pred, stats):
    """Calculate sentiment analysis metrics"""
    if not y_true or not y_pred:
        return set_default_binary_metrics(stats)
        
    y_true_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_true]
    y_pred_bin = [1 if x in ['positive', '1', 'yes'] else 0 for x in y_pred]

    stats["accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
    stats["precision"] = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    stats["recall"] = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    stats["f1_score"] = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)

    cm = confusion_matrix(y_true_bin, y_pred_bin)
    if cm.size == 0:
        return set_default_binary_metrics(stats)
    
    try:
        tn, fp, fn, tp = cm.ravel()
        stats["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    except ValueError:
        return set_default_binary_metrics(stats)

    return stats


def calculate_medical_type_metrics(y_true, y_pred, stats):
    """Calculate medical type metrics (multi-class)"""
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

    return stats


def calculate_ecqa_metrics_type(y_true, y_pred, stats):
    """Calculate ECQA metrics (multi-class)"""
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

    return stats


def calculate_snarks_metrics_type(y_true, y_pred, stats):
    """Calculate SNARKS metrics"""
    def snarks_to_bin(x):
        if x is None:
            return None
        t = str(x).strip().upper()
        t = t.replace('(', '').replace(')', '').replace('.', '').replace(' ', '')
        t = re.sub(r'^(CHOICE|OPTION|ANSWER)', '', t)
        return 1 if t == 'A' else 0 if t == 'B' else None

    y_true_bin_raw = [snarks_to_bin(x) for x in y_true]
    y_pred_bin_raw = [snarks_to_bin(x) for x in y_pred]

    pairs = [(t, p) for t, p in zip(y_true_bin_raw, y_pred_bin_raw) if t is not None and p is not None]

    if not pairs:
        stats = set_default_binary_metrics(stats)
    else:
        y_true_bin, y_pred_bin = map(list, zip(*pairs))
        stats = calculate_binary_metrics(y_true_bin, y_pred_bin, stats)

    return stats


def calculate_hotel_metrics_type(y_true, y_pred, stats):
    """Calculate hotel metrics"""
    def hotel_to_bin(x):
        if x is None:
            return None
        t = str(x).strip().lower()
        t = re.sub(r"[^a-z]", "", t)
        return 1 if t == "truthful" else 0 if t == "deceptive" else None

    y_true_bin_raw = [hotel_to_bin(x) for x in y_true]
    y_pred_bin_raw = [hotel_to_bin(x) for x in y_pred]

    pairs = [(t, p) for t, p in zip(y_true_bin_raw, y_pred_bin_raw) if t is not None and p is not None]

    if not pairs:
        stats = set_default_binary_metrics(stats)
    else:
        y_true_bin, y_pred_bin = map(list, zip(*pairs))
        stats = calculate_binary_metrics(y_true_bin, y_pred_bin, stats)

    return stats


def calculate_legal_metrics_type(y_true, y_pred, stats):
    """Calculate legal metrics"""
    labels = ["0", "1", "2", "3", "4"]
    
    y_true_clean = [str(x).strip() for x in y_true if str(x).strip() in labels]
    y_pred_clean = [str(x).strip() for x in y_pred if str(x).strip() in labels]
    
    min_len = min(len(y_true_clean), len(y_pred_clean))
    if min_len == 0:
        return set_default_legal_metrics(stats)
    
    y_true_clean = y_true_clean[:min_len]
    y_pred_clean = y_pred_clean[:min_len]

    stats["accuracy"] = accuracy_score(y_true_clean, y_pred_clean)
    
    correct = sum(1 for true, pred in zip(y_true_clean, y_pred_clean) if true == pred)
    incorrect = len(y_true_clean) - correct
    
    stats["correct"] = correct
    stats["incorrect"] = incorrect

    return stats


def set_default_legal_metrics(stats):
    """Set default values for legal metrics when no valid pairs exist"""
    stats["accuracy"] = 0.0
    stats["correct"] = 0
    stats["incorrect"] = 0
    return stats


def calculate_binary_metrics(y_true, y_pred, stats):
    """Calculate binary classification metrics"""
    stats["accuracy"] = accuracy_score(y_true, y_pred)
    stats["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    stats["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    stats["f1_score"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    stats["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    return stats


def set_default_binary_metrics(stats):
    """Set default values for binary metrics when no valid pairs exist"""
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
    return stats


def count_predictions(y_true, y_pred, data_type, stats):
    """Count predictions for statistics"""
    if data_type == "medical":
        stats["yes"] = sum(1 for x in y_pred if x == "yes")
        stats["no"] = sum(1 for x in y_pred if x == "no")
        stats["maybe"] = sum(1 for x in y_pred if x == "maybe")
    elif data_type == "sentiment":
        stats["positive"] = sum(1 for x in y_pred if x == "positive")
        stats["negative"] = sum(1 for x in y_pred if x == "negative")
    elif data_type == "ecqa":
        from collections import Counter
        pred_counts = Counter(y_pred)
        for label in set(y_pred):
            stats[label] = pred_counts[label]
    elif data_type == "snarks":
        stats["(A)"] = int(sum(1 for x in y_pred if str(x).strip().upper() in ["A", "(A)"]))
        stats["(B)"] = int(sum(1 for x in y_pred if str(x).strip().upper() in ["B", "(B)"]))
    elif data_type == "hotel":
        stats["deceptive"] = int(sum(1 for x in y_pred if str(x).strip().lower() == "deceptive"))
        stats["truthful"] = int(sum(1 for x in y_pred if str(x).strip().lower() == "truthful"))
    elif data_type == "legal":
        for i in range(5):
            stats[f"holding_{i}"] = int(sum(1 for x in y_pred if str(x).strip() == str(i)))
    else:
        stats["(A)"] = int(sum(1 for x in y_pred if str(x).strip().upper() in ["A", "(A)", "YES", "POSITIVE"]))
        stats["(B)"] = int(sum(1 for x in y_pred if str(x).strip().upper() in ["B", "(B)", "NO", "NEGATIVE"]))

    return stats


def store_classification_results(dataset_id, user_id, method, provider, model_name, data_type, results, stats):
    """Store classification results in database"""
    classification_data = {
        "dataset_id": ObjectId(dataset_id),
        "user_id": ObjectId(user_id),
        "method": method,
        "provider": provider if method == 'llm' else None,
        "model": model_name if method == 'llm' else None,
        "data_type": data_type,
        "results": results,
        "created_at": datetime.now(),
        "stats": stats,
        "classification_type": "classification_only"  # Mark as classification-only
    }

    classification_id = mongo.db.classifications.insert_one(classification_data).inserted_id
    return classification_id
