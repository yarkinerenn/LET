# ---------- /api/classify_and_explain/<dataset_id> ----------
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

classify_and_explain_bp = Blueprint("classif_and_explain", __name__)
CLASSIFICATION_METHODS = ['bert', 'llm']




@classify_and_explain_bp.route('/api/classify_and_explain/<dataset_id>', methods=['POST'])
@login_required
def classify_and_explain(dataset_id):
    try:
        # Extract and validate request data
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Process the classification request
        result = process_classification_request(dataset_id, data, current_user)
        return jsonify(result), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Classification+explanation failed", "details": str(e)}), 500


def process_classification_request(dataset_id, data, current_user):
    """Main function to process classification requests"""
    # Validate method
    method = data.get('method')
    if method not in CLASSIFICATION_METHODS:
        raise ValueError(f"Invalid method. Must be one of: {CLASSIFICATION_METHODS}")

    # Get data type and column mappings
    data_type = data.get('dataType', 'sentiment')
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

    # Process each sample
    results, explanations_to_save, stats = process_samples(
        df_sampled, method, data_type, text_column, label_column,
        client, provider, model_name
    )

    # Calculate metrics
    stats = calculate_metrics(results, data_type, stats)

    # Store results in database
    classification_id = store_classification_results(
        dataset_id, current_user.id, method, provider,
        model_name, data_type, results, stats
    )

    # Save explanations
    save_explanations(classification_id, current_user.id, explanations_to_save, results)

    return {
        "message": "Classification+explanation completed",
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
    """Initialize and return a chat-completions compatible client for the given provider.
    Supports: openai, groq, deepseek, ollama, openrouter, gemini.
    Returns a client object or None (for providers that don't require a client here).
    """
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
        # Deepseek uses OpenAI-compatible API with a custom base_url
        return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    elif prov == 'ollama':
        # For Ollama we call a separate path (get_ollama_response) and don't need a client here.
        # Returning None keeps downstream logic unchanged.
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
        # Gemini's OpenAI-compatible endpoint
        return OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=api_key)

    else:
        raise ValueError("Invalid LLM provider")



def sample_dataset(df, data_type, label_column, sample_size):
    """Sample the dataset based on data type requirements"""
    if data_type in ["ecqa", "snarks"]:
        # For ECQA and SNARKS, don't stratify, just sample N entries
        return df.sample(n=sample_size, random_state=42)

    elif data_type == "hotel":
        # Stratify by both gold label and polarity if available
        strat_cols = [label_column, "polarity"]
        if all(c in df.columns for c in strat_cols):
            strata = df[strat_cols].astype(str).agg("||".join, axis=1)
            try:
                df_sampled, _ = train_test_split(
                    df, train_size=sample_size, stratify=strata, random_state=42
                )
                return df_sampled
            except ValueError:
                # Fallback if some strata have too few samples
                return df.sample(n=sample_size, random_state=42)

    elif label_column in df.columns:
        # Standard stratified sampling
        df_sampled, _ = train_test_split(
            df, train_size=sample_size, stratify=df[label_column], random_state=42
        )
        return df_sampled

    # Default sampling
    return df.sample(n=sample_size, random_state=42)


def process_samples(df_sampled, method, data_type, text_column, label_column, client, provider, model_name):
    """Process all samples in the dataset"""
    results = []
    explanations_to_save = []
    stats = initialize_stats(data_type)

    for df_idx, row in df_sampled.iterrows():
        try:
            result, explanation = process_single_sample(
                row, df_idx, method, data_type, text_column, label_column,
                client, provider, model_name
            )

            if result:
                results.append(result)
                if explanation:
                    explanations_to_save.append(explanation)

                # Update statistics
                update_stats(stats, data_type, result)

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing row: {str(e)}")
            continue

    return results, explanations_to_save, stats


def initialize_stats(data_type):
    """Initialize statistics based on data type"""
    stats = {"total": 0}

    if data_type == "sentiment":
        stats.update({"positive": 0, "negative": 0})
    elif data_type == "legal":
        stats.update({"correct": 0, "incorrect": 0})
    elif data_type == "medical":
        stats.update({"correct": 0, "incorrect": 0, "maybe": 0})
    elif data_type == "hotel":
        stats.update({"correct": 0, "incorrect": 0})
    # For ECQA, we'll update stats dynamically based on labels

    return stats


def process_single_sample(row, df_idx, method, data_type, text_column, label_column, client, provider, model_name):
    """Process a single sample row"""
    # Generate appropriate prompt
    prompt = generate_prompt(data_type, row, text_column, provider)

    # Get LLM response if using LLM method
    if method == 'llm':
        label, score, explanation, content = get_llm_response(
            client, provider, model_name, prompt, data_type
        )

        # Save explanation for later storage
        explanation_data = {
            "df_index": int(df_idx),
            "explanation": explanation,
            "model": model_name,
            "type": "llm",
        }
    else:
        label, score, explanation, content = None, 0.0, "", ""
        explanation_data = None

    # Prepare result data based on data type
    result_data = prepare_result_data(
        data_type, row, label_column, label, score, explanation, df_idx, content
    )

    # Calculate additional metrics for certain data types
    if data_type in ["medical", "ecqa", "snarks", "hotel"]:
        result_data = calculate_additional_metrics(
            data_type, result_data, provider, model_name
        )

    return result_data, explanation_data


def generate_prompt(data_type, row, text_column, provider):
    """Generate appropriate prompt based on data type"""
    
    if data_type == "sentiment":
        text = str(row[text_column])
        return f"""Given the text below, classify the sentiment as either POSITIVE or NEGATIVE, and briefly explain your reasoning in 2-3 sentences.

Text: {text}

Format your answer as:
Sentiment: <POSITIVE/NEGATIVE>
Explanation: <your explanation here>
"""

    elif data_type == "legal":
        context = str(row[text_column])
        choices = [row.get(f'holding_{i}', '') for i in range(5)]
        holdings_str = "\n".join([f"{i}: {c}" for i, c in enumerate(choices)])
        return f"""Assume you are a legal advisor

        Statement: {context}
        Holdings:
        {holdings_str}
        select the most appropriate holding (choose 0, 1, 2, 3, or 4) and explain your recommendation.
        
        Format your answer as:
        Holding: <number>
        Explanation: <your explanation here>
        """

    elif data_type == "medical":
        question = str(row.get("question", ""))
        context = pretty_pubmed_qa(row.get("context", ""))
        format_text = "Format your answer as:\nAnswer: <yes/no/maybe>\nExplanation: <your explanation here>" if provider != 'ollama' else ""
        return f"""Assume you are a Medical advisor 

Question: {question}
Context: {context}

Answer the questions with Yes/No/maybe and give an explanation for your recommendation.

{format_text}
"""

    elif data_type == "ecqa":
        question = str(row[text_column])
        choices = [row.get('q_op1', ''), row.get('q_op2', ''), row.get('q_op3', ''), row.get('q_op4', ''), row.get('q_op5', '')]
        return f"""Given the following question and five answer options, select the best answer and explain your choice in 2-3 sentences. YOU MUST ONLY CHOOSE ONE OF THE CHOICES

Question: {question}

Choices:
1. {choices[0]}
2. {choices[1]}
3. {choices[2]}
4. {choices[3]}
5. {choices[4]}

Format your answer as:
Answer: <Choice as number>
Explanation: <your explanation here>
"""

    elif data_type == "snarks":
        question = str(row[text_column])
        return f"""You are a sarcasm detection system. You will chose (A) or (B) as your answer and explain your decision in 2-3 sentences. Do not quote from the question or mention any words in your explanation.

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
        return f"""You are a deceptive hotel review detection system. You will chose "truthful" or "deceptive" as your answer and explain your decision in 2-3 sentences.

Question: {question}

Format your answer as:
Answer: <Choice as "truthful" or "deceptive">
Explanation: <your explanation here>
"""
    
    return ""


def get_llm_response(client, provider, model_name, prompt, data_type):
    """Get response from LLM based on provider"""
    if provider == "ollama":
        return get_ollama_response(prompt, data_type, model_name)
    else:
        return get_standard_llm_response(client, model_name, prompt, data_type)


def get_standard_llm_response(client, model_name, prompt, data_type):
    """Get response from standard LLM APIs (OpenAI, Groq, etc.)"""
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    print(content, 'this is what llm prompt')

    return parse_llm_response(content, data_type)


def get_ollama_response(prompt, data_type, model_name):
    """Get response from Ollama with special handling"""
    llm = Ollama(model=model_name, temperature=0)
    content = llm.invoke(prompt)
    print(content, 'this is what ollama prompt')

    # For Ollama, we might need additional processing
    # This is a placeholder for any Ollama-specific processing

    return parse_llm_response(content, data_type)


def parse_llm_response(content, data_type):
    """Parse LLM response based on data type"""
    
    if data_type == "sentiment":
        # Parse sentiment analysis response
        m = re.search(r"Sentiment:\s*(POSITIVE|NEGATIVE)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if m:
            label = m.group(1).upper()
            explanation = m.group(2).strip()
        else:
            lines = content.split('\n')
            label = lines[0].replace("Sentiment:", "").strip().upper()
            explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()
    
    elif data_type == "legal":
        # Parse legal analysis response
        m = re.search(r"Holding:\s*([0-4])[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if m:
            label = int(m.group(1))
            explanation = m.group(2).strip()
        else:
            num_match = re.search(r"[0-4]", content)
            label = int(num_match.group(0)) if num_match else -1
            explanation = content
    
    elif data_type == "medical":
        # Parse medical analysis response
        m = re.search(r"Answer:\s*(yes|no|maybe)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if m:
            label = m.group(1).lower()
            explanation = m.group(2).strip()
        else:
            lines = content.split('\n')
            label = lines[0].replace("Answer:", "").strip().lower()
            explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()
    
    else:  # ecqa, snarks, hotel - general format
        # Parse general response format: Answer: ... Explanation: ...
        m = re.search(r"Answer:\s*(.+)[\s\n]+Explanation:\s*(.*)", content, re.IGNORECASE | re.DOTALL)
        if m:
            label = m.group(1).strip()
            explanation = m.group(2).strip()
        else:
            lines = content.split('\n')
            label = lines[0].replace("Answer:", "").strip()
            explanation = "\n".join(lines[1:]).replace("Explanation:", "").strip()

    return label, 1.0, explanation, content


def prepare_result_data(data_type, row, label_column, label, score, explanation, df_idx, content):
    """Prepare result data based on data type"""
    
    # Base result data
    result_data = {
        "label": label,
        "score": score,
        "llm_explanation": explanation,
        "df_index": int(df_idx)
    }
    
    # Add ground truth if present
    if label_column:
        result_data["actualLabel"] = row[label_column]
    
    # Add data type specific fields
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
            "ground_explanation": row.get("taskB", "")
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
            "choices": [row.get(f'holding_{i}', '') for i in range(5)]
        })
    else:
        # Default case - include original data
        result_data["original_data"] = row.to_dict()

    return result_data


def calculate_additional_metrics(data_type, result_data, provider, model_name):
    """Calculate additional metrics for specific data types"""
    if data_type not in ["medical", "ecqa", "snarks", "hotel"]:
        return result_data
    
    # Get API keys
    groq_key = get_user_api_key_groq()
    api_key = get_api_key_for_provider(provider)
    
    # Prepare common row reference
    row_reference = {
        "ground_question": result_data["question"],
        "ground_label": result_data["actualLabel"],
        "predicted_explanation": result_data["llm_explanation"],
        "predicted_label": result_data["label"],
    }
    
    if data_type == "medical":
        # Initialize NER pipeline for medical
        ner_pipe = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
        row_reference.update({
            "ground_explanation": result_data["long_answer"],
            "ground_context": result_data["context"],
        })
        
        # Calculate trust score
        
        trust_score = lext(
            result_data["context"], result_data["question"], result_data["long_answer"],
            result_data["actualLabel"], model_name, groq_key, provider, api_key,
            ner_pipe, "medical", row_reference
        )
        
        # Extract metrics
        result_data["metrics"] = {
            "plausibility_metrics": {
                "iterative_stability": row_reference.get("iterative_stability"),
                "paraphrase_stability": row_reference.get("paraphrase_stability"),
                "consistency": row_reference.get("consistency"),
                "plausibility": row_reference.get("plausibility")
            },
            "faithfulness_metrics": {
                "qag_score": row_reference.get("qag_score"),
                "counterfactual": row_reference.get("counterfactual_scaled"),
                "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                "faithfulness": row_reference.get("faithfulness")
            },
            "trustworthiness_score": row_reference.get("trustworthiness"),
            "lext_score": trust_score
        }
    
    elif data_type == "ecqa":
        row_reference["ground_explanation"] = result_data["ground_explanation"]
        
        # Calculate trust score
        trust_score = lext(
            None, result_data["question"], result_data["ground_explanation"],
            result_data["actualLabel"], model_name, groq_key, provider, api_key,
            None, "ecqa", row_reference
        )
        
        result_data["metrics"] = {
            "plausibility_metrics": {
                "correctness": row_reference.get("correctness"),
                "consistency": row_reference.get("consistency"),
                "plausibility": row_reference.get("plausibility")
            },
            "faithfulness_metrics": {
                "qag_score": row_reference.get("qag_score"),
                "counterfactual": row_reference.get("counterfactual_scaled"),
                "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                "faithfulness": row_reference.get("faithfulness")
            },
            "trustworthiness_score": row_reference.get("trustworthiness"),
            "lext_score": trust_score
        }
    
    elif data_type in ["snarks", "hotel"]:
        row_reference["ground_explanation"] = None
        
        # Calculate faithfulness score
        faithfulness_score = faithfulness(
            result_data["llm_explanation"], result_data["label"], result_data["question"],
            result_data["actualLabel"], None, groq_key, model_name, provider, api_key,
            data_type, row_reference
        )
        
        result_data["metrics"] = {
            "faithfulness_metrics": {
                "qag_score": row_reference.get("qag_score"),
                "counterfactual": row_reference.get("counterfactual_scaled"),
                "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
                "faithfulness": faithfulness_score
            }
        }
    
    return result_data


def get_api_key_for_provider(provider):
    """Get API key for the specified provider"""
    if provider == "openrouter":
        return get_user_api_key_openrouter()
    elif provider == "openai":
        return get_user_api_key_openai()
    elif provider == "groq":
        return get_user_api_key_groq()
    elif provider == "gemini":
        return get_user_api_key_gemini()
    return 'api'  # Default


def update_stats(stats, data_type, result):
    """Update statistics based on result"""
    stats["total"] += 1

    if data_type == "sentiment":
        if result["label"] == "POSITIVE":
            stats["positive"] += 1
        else:
            stats["negative"] += 1

    elif data_type in ["legal", "medical", "hotel"]:
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
    else:
        # Default binary classification metrics
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

    if 'original_data' in result:
        original = result['original_data']
        if 'final_decision' in original:
            return str(original['final_decision']).strip().lower()
        elif 'label' in original:
            return str(original['label']).strip().lower()

    return None


def calculate_sentiment_metrics(y_true, y_pred, stats):
    """Calculate sentiment analysis metrics"""
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

    # Keep only valid pairs
    pairs = [(t, p) for t, p in zip(y_true_bin_raw, y_pred_bin_raw) if t is not None and p is not None]

    if not pairs:
        # No valid pairs
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

    # Keep only valid pairs
    pairs = [(t, p) for t, p in zip(y_true_bin_raw, y_pred_bin_raw) if t is not None and p is not None]

    if not pairs:
        # No valid pairs
        stats = set_default_binary_metrics(stats)
    else:
        y_true_bin, y_pred_bin = map(list, zip(*pairs))
        stats = calculate_binary_metrics(y_true_bin, y_pred_bin, stats)

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
        "stats": stats
    }

    classification_id = mongo.db.classifications.insert_one(classification_data).inserted_id

    # Update with explanation models
    explanation_models = [{'provider': provider, 'model': model_name.replace('.', '_')}]
    mongo.db.classifications.update_one(
        {"_id": ObjectId(classification_id), "user_id": ObjectId(user_id)},
        {"$set": {"explanation_models": explanation_models, "updated_at": datetime.now()}}
    )

    return classification_id


def save_explanations(classification_id, user_id, explanations_to_save, results):
    """Save explanations to database"""
    for explanation_entry in explanations_to_save:
        # Find the corresponding result index by matching df_index
        result_id = next(
            (i for i, r in enumerate(results) if r.get("df_index") == explanation_entry["df_index"]),
            None
        )
        if result_id is not None:
            save_explanation_to_db(
                classification_id=str(classification_id),
                user_id=user_id,
                result_id=result_id,
                explanation_type=explanation_entry["type"],
                content=explanation_entry["explanation"],
                model_id=explanation_entry["model"].replace('.', '_')
            )