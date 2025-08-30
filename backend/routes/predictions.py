# routes/predictions.py
from flask import Blueprint, jsonify, current_app, request
from flask_login import login_required, current_user
from bson import ObjectId
from extensions import mongo

predictions_bp = Blueprint("predictions", __name__)

@predictions_bp.route('/api/predictions/update_prediction_label/<prediction_id>', methods=['POST'])
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



@predictions_bp.route('/api/delete_prediction/<classification_id>', methods=['DELETE'])
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

@predictions_bp.route('/api/predictions/<prediction_id>', methods=['GET'])
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
            "model": prediction["model"],
            "id": str(prediction["_id"]),
            "text": prediction["text"],
            "label": prediction["label"],
            "confidence": prediction["score"],
            "timestamp": prediction["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "user_label": prediction.get("user_label")  # safer .get
        }

        return jsonify({"classification": result})

    except Exception as e:
        current_app.logger.exception(f"Error fetching prediction: {e}")
        return jsonify({"error": str(e)}), 500


@predictions_bp.route('/api/predictions', methods=['GET'])
@login_required
def get_predictions():
    """Get the last 50 classifications made from the dashboard independent of any dataset"""
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

        results = []
        for prediction in predictions:
            results.append({
                "model": prediction["model"],
                "id": str(prediction["_id"]),
                "text": prediction["text"],
                "label": prediction["label"],
                "score": prediction["score"],
                "timestamp": prediction["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify({"classifications": results})

    except Exception as e:
        current_app.logger.exception(f"Error fetching predictions: {e}")
        return jsonify({"error": str(e)}), 500