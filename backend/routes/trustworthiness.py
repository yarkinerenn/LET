# routes/trustworthiness.py
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from bson import ObjectId

# external libs you already use
from transformers import pipeline

# your project modules
from extensions import mongo  # not used here, but handy if you log to DB later
from LExT.metrics.trustworthiness import lext
from LExT.metrics.faithfulness import faithfulness

# pull API key getters from your auth blueprint module
from .auth import (
    get_user_api_key_openai,
    get_user_api_key_openrouter,
    get_user_api_key_groq,
    get_user_api_key_deepseek_api,
    get_user_api_key_gemini,
)

# if you already have this util, import it; else this safe fallback keeps your code running
try:
    from helpers import extract_context_explanation
except Exception:
    def extract_context_explanation(ctx):
        return ctx

trust_bp = Blueprint("trust", __name__)


def _resolve_api_key(provider: str) -> str | None:
    """Return the API key string based on provider name."""
    if provider == "openrouter":
        return get_user_api_key_openrouter()
    if provider == "openai":
        return get_user_api_key_openai()
    if provider == "groq":
        return get_user_api_key_groq()
    if provider == "deepseek":
        return get_user_api_key_deepseek_api()
    if provider == "gemini":
        return get_user_api_key_gemini()
    return None


@trust_bp.route('/api/trustworthiness', methods=['POST'])
@login_required
def trustworthiness_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        current_app.logger.debug({"trustworthiness_payload": data})

        # Extract fields from request
        question = data.get("ground_question")
        ground_explanation = data.get("ground_explanation")
        ground_label = data.get("ground_label")
        explanation = data.get("predicted_explanation")
        label = data.get("predicted_label")

        # Optional context
        context = data.get("ground_context", None)
        context = extract_context_explanation(context)

        # Model/provider selection
        provider = data.get("provider")
        api = _resolve_api_key(provider) or "api"
        target_model = data.get("target_model")
        data_type = data.get("data_type", "medical")  # Default to medical
        labels = data.get("labels", [])  # Labels for the dataset

        # NER pipeline used by your lext implementation (only for medical)
        ner_pipe = None
        if data_type == "medical":
            ner_pipe = pipeline(
                "token-classification",
                model="Clinical-AI-Apollo/Medical-NER",
                aggregation_strategy="simple"
            )

        # Row reference to collect intermediate metrics
        row_reference = {
            "ground_question": question,
            "ground_explanation": ground_explanation,
            "ground_label": ground_label,
            "predicted_explanation": explanation,
            "predicted_label": label,
            "ground_context": context,
        }

        # Get Groq API key
        groq_key = get_user_api_key_groq()

        # Compute trustworthiness (LExT) - Updated signature to match classify_and_explain.py
        score = lext(
            context,
            question,
            ground_explanation,
            ground_label,
            target_model,
            groq_key,
            provider,
            api,
            ner_pipe,
            data_type,
            row_reference,
            labels
        )

        # Optionally expose sub-metrics if your frontend needs them
        plausibility_metrics = {
            "iterative_stability": row_reference.get("iterative_stability"),
            "paraphrase_stability": row_reference.get("paraphrase_stability"),
            "consistency": row_reference.get("consistency"),
            "plausibility": row_reference.get("plausibility"),
        }
        faithfulness_metrics = {
            "qag_score": row_reference.get("qag_score"),
            "counterfactual": row_reference.get("counterfactual_scaled"),
            "contextual_faithfulness": row_reference.get("contextual_faithfulness"),
            "faithfulness": row_reference.get("faithfulness"),
        }
        trustworthiness_score = row_reference.get("trustworthiness", score)

        return jsonify({
            "trustworthiness_score": score,
            "metrics": {
                "plausibility_metrics": plausibility_metrics,
                "faithfulness_metrics": faithfulness_metrics,
                "trustworthiness_score": trustworthiness_score
            }
        })

    except Exception as e:
        current_app.logger.exception(f"/api/trustworthiness error: {e}")
        return jsonify({"error": "Server error"}), 500


@trust_bp.route('/api/faithfulness', methods=['POST'])
@login_required
def faithfulness_endpoint():
    try:
        data = request.get_json(silent=True) or {}
        current_app.logger.debug({"faithfulness_payload": data})

        # Extract fields from request
        question = data.get("ground_question")
        ground_explanation = data.get("ground_explanation")
        ground_label = data.get("ground_label")
        explanation = data.get("predicted_explanation")
        label = data.get("predicted_label")

        # Optional context
        context = data.get("ground_context", None)
        if context:
            context = extract_context_explanation(context)

        # Provider/api
        provider = data.get("provider")
        api = _resolve_api_key(provider) or "api"
        target_model = data.get("target_model", "llama3:8b")
        data_type = data.get("data_type", "sentiment")  # Default to sentiment
        labels = data.get("labels", [])  # Labels for the dataset

        # Row reference for diagnostics
        row_reference = {
            "ground_question": question,
            "ground_explanation": ground_explanation,
            "ground_label": ground_label,
            "predicted_explanation": explanation,
            "predicted_label": label,
            "ground_context": context,
        }

        # Get Groq API key
        groq_key = get_user_api_key_groq()

        # Compute faithfulness score - Updated signature to match classify_and_explain.py
        score = faithfulness(
            explanation,
            label,
            question,
            ground_label,
            context,
            groq_key,
            target_model,
            provider,
            api,
            data_type,
            labels,
            row_reference
        )

        return jsonify({
            "faithfulness_score": score,
            "details": row_reference
        })

    except Exception as e:
        current_app.logger.exception(f"/api/faithfulness error: {e}")
        return jsonify({"error": "Server error"}), 500