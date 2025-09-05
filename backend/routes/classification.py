# routes/classification.py
import traceback
from datetime import datetime

import pandas as pd
from flask import Blueprint, jsonify, current_app
from flask_login import login_required, current_user
from bson import ObjectId

from extensions import mongo

classification_bp = Blueprint("classification", __name__)


@classification_bp.route('/api/classification/<classification_id>', methods=['GET'])
@login_required
def get_classification_details(classification_id):
    """Get the classification details to see previous classification on the detailed dataset view page."""
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
        current_app.logger.exception(f"get_classification_details error: {e}")
        return jsonify({"error": str(e)}), 500


@classification_bp.route('/api/classification/stats/<classification_id>', methods=['GET'])
@login_required
def get_classification_stats(classification_id):
    """Get the classification statistics."""
    try:
        classification = mongo.db.classifications.find_one({
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(current_user.id)
        }, {"stats": 1, "method": 1, "model": 1, "provider": 1})

        if not classification:
            return jsonify({"error": "Classification not found"}), 404

        # Calculate additional metrics (defaults if missing)
        stats = classification.get("stats", {}) or {}
        stats.update({
            "accuracy": stats.get("accuracy", 0),
            "precision": stats.get("precision", 0),
            "recall": stats.get("recall", 0),
            "f1_score": stats.get("f1_score", 0),
        })

        return jsonify({
            "method": classification.get("method"),
            "model": classification.get("model"),
            "provider": classification.get("provider"),
            "stats": stats
        }), 200

    except Exception as e:
        current_app.logger.exception(f"get_classification_stats error: {e}")
        return jsonify({"error": str(e)}), 500


@classification_bp.route('/api/classifications/<dataset_id>', methods=['GET'])
@login_required
def get_dataset_classifications(dataset_id):
    """List classifications for a given dataset for the current user."""
    try:
        classifications = list(mongo.db.classifications.find(
            {
                "dataset_id": ObjectId(dataset_id),
                "user_id": ObjectId(current_user.id)
            },
            {
                "method": 1,
                "provider": 1,
                "model": 1,
                "created_at": 1,
                "stats": 1
            }
        ).sort("created_at", -1))

        for cls in classifications:
            cls['_id'] = str(cls['_id'])
            cls['created_at'] = cls['created_at'].strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({"classifications": classifications}), 200

    except Exception as e:
        current_app.logger.exception(f"get_dataset_classifications error: {e}")
        return jsonify({"error": str(e)}), 500


@classification_bp.route('/api/delete_classification/<classification_id>', methods=['DELETE'])
@login_required
def delete_classification(classification_id):
    """Delete a classification that is made for a whole dataset."""
    try:
        result = mongo.db.classifications.delete_one({
            '_id': ObjectId(classification_id),
            'user_id': ObjectId(current_user.id)  # Ensure user can only delete their own
        })

        if result.deleted_count == 0:
            return jsonify({"error": "Classification not found or unauthorized"}), 404

        return jsonify({"message": "Classification deleted successfully"}), 200

    except Exception as e:
        current_app.logger.exception(f"delete_classification error: {e}")
        return jsonify({"error": str(e)}), 500


@classification_bp.route('/api/classificationentry/<classification_id>/<result_id>', methods=['GET'])
@login_required
def get_classificationentry(classification_id, result_id):
    """Get a single entry of a classified dataset (sentiment, legal, medical, ECQA, etc)."""
    try:
        classification = mongo.db.classifications.find_one({
            "_id": ObjectId(classification_id),
            "user_id": ObjectId(current_user.id)
        })

        if not classification:
            return jsonify({"error": "Classification not found"}), 404

        result = classification['results'][int(result_id)]
        data_type = classification.get('data_type')

        # Default/Universal fields
        response_data = {
            "prediction": result.get('label', ''),
            "confidence": result.get('score', ''),
            "actualLabel": result.get('actualLabel', ''),
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
            "data_type": data_type,
            "method": classification.get('method'),
        }

        # Per-type enrichment
        if data_type == "legal":
            holdings = []
            for i in range(5):
                holding_key = f'holding_{i}'
                if holding_key in result.get('original_data', {}):
                    holdings.append(result['original_data'][holding_key])
            response_data.update({
                "text": result.get('citing_prompt', ''),
                "holdings": holdings,
            })

        elif data_type == "medical":
            response_data.update({
                "text": result.get('question', ''),
                "question": result.get('question', ''),
                "context": result.get('context', ''),
                "long_answer": result.get('long_answer', ''),
                "trustworthiness_score": result.get("metrics", {}).get("trustworthiness_score"),
                "plausibility_score": result.get("metrics", {}).get("plausibility_metrics", {}).get("plausibility"),
                "faithfulness_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("faithfulness"),
            })

        elif data_type == "sentiment":
            response_data.update({
                "text": result.get('text', ''),
                 "faithfulness_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("faithfulness"),
                "qag_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("qag_score"),
                "counterfactual": result.get("metrics", {}).get("faithfulness_metrics", {}).get("counterfactual"),
                "contextual_faithfulness": result.get("metrics", {}).get("faithfulness_metrics", {}).get("contextual_faithfulness"),
            })

        elif data_type == "ecqa":
            response_data.update({
                "question": result.get('question', ''),
                "choices": result.get('choices', []),
                "ground_explanation": result.get('ground_explanation', ''),
                "text": result.get('question', ''),
                "trustworthiness_score": result.get("metrics", {}).get("trustworthiness_score"),
                "plausibility_score": result.get("metrics", {}).get("plausibility_metrics", {}).get("plausibility"),
                "correctness": result.get("metrics", {}).get("plausibility_metrics", {}).get("correctness"),
                "consistency": result.get("metrics", {}).get("plausibility_metrics", {}).get("consistency"),
                "faithfulness_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("faithfulness"),
                "qag_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("qag_score"),
                "counterfactual": result.get("metrics", {}).get("faithfulness_metrics", {}).get("counterfactual"),
                "contextual_faithfulness": result.get("metrics", {}).get("faithfulness_metrics", {}).get("contextual_faithfulness"),
            })

        elif data_type in ("snarks", "hotel"):
            response_data.update({
                "question": result.get('question', ''),
                "text": result.get('question', ''),
                "label": result.get('label', ''),
                "faithfulness_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("faithfulness"),
                "qag_score": result.get("metrics", {}).get("faithfulness_metrics", {}).get("qag_score"),
                "counterfactual": result.get("metrics", {}).get("faithfulness_metrics", {}).get("counterfactual"),
                "contextual_faithfulness": result.get("metrics", {}).get("faithfulness_metrics", {}).get("contextual_faithfulness"),
            })

        else:
            # fallback for unknown types
            response_data["text"] = result.get('text', '') or result.get('question', '') or ''

        return jsonify(response_data), 200

    except IndexError:
        return jsonify({"error": "Result not found"}), 404
    except Exception as e:
        current_app.logger.exception(f"get_classificationentry error: {e}")
        return jsonify({"error": str(e)}), 500


@classification_bp.route('/api/classification/empty/<dataset_id>', methods=['POST'])
@login_required
def create_or_get_empty_classification(dataset_id):
    """
    Create an empty/manual classification for this user/dataset if not exists, otherwise return existing one.
    Handles all supported data types automatically (sentiment, legal, medical).
    """
    user_doc = mongo.db.users.find_one({"_id": ObjectId(current_user.id)})

    MAX_RESULTS = 200  # <- Limit results to avoid a BSON document too large
    provider = user_doc.get('preferred_provider', 'openai')
    model_name = user_doc.get('preferred_model', 'gpt-3.5-turbo')
    try:
        print(dataset_id,'this is datasetid')
        # Check if one already exists
        classification = mongo.db.classifications.find_one({
            "dataset_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id),
            "method": "explore"
        })

        if classification:
            return jsonify({"classification_id": str(classification["_id"]), "already_exists": True}), 200

        # Otherwise, create it
        dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})
        if not dataset:
            print('dataset not found')
            return jsonify({"error": "Dataset not found"}), 404

        df = pd.read_csv(dataset["filepath"])

        # --- Detect type ---
        if "final_decision" in df.columns:
            data_type = "medical"
        elif "citing_prompt" in df.columns:
            data_type = "legal"
        elif "text" in df.columns:
            data_type = "sentiment"
        else:
            data_type = "unknown"

        results = []
        # Only take the first MAX_RESULTS rows!
        for _, row in df.head(MAX_RESULTS).iterrows():
            if data_type == "medical":
                results.append({
                    "question": row.get("question", ""),
                    "context": row.get("context", ""),
                    "long_answer": row.get("long_answer", ""),
                    "label": "",
                    "score": None,
                    "actualLabel": row.get("final_decision", ""),
                    "llm_explanations": {},
                    "shap_plot_explanation": "",
                    "shapwithllm_explanations": {},
                    "ratings": {},
                    "trustworthiness_score": None
                })
            elif data_type == "legal":
                results.append({
                    "citing_prompt": row.get("citing_prompt", ""),
                    "choices": [row.get(f"holding_{i}", "") for i in range(5)],
                    "label": "",
                    "score": None,
                    "actualLabel": row.get("label", ""),
                    "llm_explanations": {},
                    "shap_plot_explanation": "",
                    "shapwithllm_explanations": {},
                    "ratings": {},
                    "trustworthiness_score": None
                })
            elif data_type == "sentiment":
                results.append({
                    "text": row.get("text", ""),
                    "label": "",
                    "score": None,
                    "actualLabel": row.get("label", ""),
                    "llm_explanations": {},
                    "shap_plot_explanation": "",
                    "shapwithllm_explanations": {},
                    "ratings": {},
                    "trustworthiness_score": None
                })
            else:
                results.append({
                    "original_data": row.to_dict(),
                    "label": "",
                    "score": None,
                    "actualLabel": "",
                    "llm_explanations": {},
                    "shap_plot_explanation": "",
                    "shapwithllm_explanations": {},
                    "ratings": {},
                    "trustworthiness_score": None
                })
        explanation_models = [{'provider': provider, 'model': model_name}]


        classification_data = {
            'explanation_models':explanation_models,
            "dataset_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id),
            "method": "explore",
            "provider": None,
            "model": None,
            "data_type": data_type,
            "results": results,
            "created_at": datetime.now(),
            "stats": {},

        }
        classification_id = mongo.db.classifications.insert_one(classification_data).inserted_id
        return jsonify({"classification_id": str(classification_id), "already_exists": False, "data_type": data_type}), 201

    except Exception as e:
        traceback.print_exc()
        print(f"Error in empty classification creation: {str(e)}")
        return jsonify({"error": str(e)}), 500