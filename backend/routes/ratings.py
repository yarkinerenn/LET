# routes/ratings.py
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user
from bson import ObjectId

from extensions import mongo
from .helpers import save_ratings_to_db
ratings_bp = Blueprint("ratings", __name__)

@ratings_bp.route('/api/save_ratings', methods=['POST'])
@login_required
def save_ratings():
    """Save ratings for a classification result."""
    try:
        data = request.get_json() or {}

        classification_id = data.get("classificationId")
        result_id = data.get("resultId")
        ratings = data.get("ratings")
        timestamp = data.get("timestamp")

        # user id from current_user
        user_id = ObjectId(current_user.id)

        if not all([classification_id, result_id, ratings, timestamp]):
            return jsonify({"success": False, "message": "Missing required fields"}), 400

        success = save_ratings_to_db(classification_id, user_id, result_id, ratings, timestamp)

        if success:
            return jsonify({"success": True}), 200
        return jsonify({"success": False, "message": "Update failed"}), 404

    except Exception as e:
        current_app.logger.exception(f"Error saving ratings: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500


@ratings_bp.route('/api/track-selection', methods=['POST'])
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