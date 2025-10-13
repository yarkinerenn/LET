# routes/datasets.py
from flask import jsonify, request, current_app,Blueprint
from flask_login import login_required, current_user
from bson import ObjectId
from datetime import datetime
from datasets import load_dataset
from .helpers import pretty_pubmed_qa
import pandas as pd
import os

from extensions import mongo,login_manager
# attach to the existing blueprint from route.py

datasets_bp = Blueprint("datasets", __name__)
@datasets_bp.route("/api/dataset/<dataset_id>", methods=["GET"])
def get_dataset(dataset_id):
    """Get the uploaded dataset with optional pagination"""

    dataset = mongo.db.datasets.find_one({"_id": ObjectId(dataset_id)})

    if not dataset:
        return jsonify({"error": "Dataset not found"}), 404

    filepath = dataset["filepath"]
    try:
        df = pd.read_csv(filepath)
        
        # Get pagination parameters from query string
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        # Calculate pagination
        total_count = len(df)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        # Get the requested page of data
        data_page = df.iloc[start_idx:end_idx].to_dict(orient="records")
        
        return jsonify({
            "filename": dataset["filename"], 
            "data": data_page,
            "total_count": total_count,
            "page": page,
            "limit": limit
        })
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/api/import_hf_dataset", methods=["POST"])
@login_required
def import_hf_dataset():
    """Import a dataset from Hugging Face to the DB (and store a CSV on disk)."""
    data = request.json or {}
    hf_dataset_name = data.get("dataset_name")
    file_name = data.get("file_name", (hf_dataset_name or "dataset") + ".csv")

    if not hf_dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    try:
        ds = load_dataset(hf_dataset_name, trust_remote_code=True)
        # prefer 'train' split if available; else use the first split
        split = "train" if "train" in ds else next(iter(ds.keys()))
        df = ds[split].to_pandas()

        # prettify context if present
        if "context" in df.columns:
            df["context"] = df["context"].apply(pretty_pubmed_qa)

        upload_dir = current_app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, file_name)

        if os.path.exists(filepath):
            os.remove(filepath)

        df.to_csv(filepath, index=False)

        dataset_entry = {
            "filename": file_name,
            "filepath": filepath,
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "HuggingFace",
            "user_id": ObjectId(current_user.id),
        }
        mongo.db.datasets.insert_one(dataset_entry)

        return jsonify({"message": "Dataset imported successfully"}), 201

    except Exception as e:
        # best-effort cleanup
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        current_app.logger.exception(f"Error importing HF dataset: {e}")
        return jsonify({"error": f"Failed to import dataset: {str(e)}"}), 500


@datasets_bp.route("/api/delete_dataset/<dataset_id>", methods=["DELETE"])
@login_required
def delete_dataset(dataset_id):
    """Delete a dataset document and, if inside UPLOAD_FOLDER, the file too."""
    try:
        dataset = mongo.db.datasets.find_one({
            "_id": ObjectId(dataset_id),
            "user_id": ObjectId(current_user.id),
        })
        if not dataset:
            return jsonify({"error": "Dataset not found or unauthorized"}), 404
        
        # Prevent deletion of public datasets
        if dataset.get("is_public", False):
            return jsonify({"error": "Public datasets cannot be deleted"}), 403

        file_path = dataset.get("filepath")

        # delete DB record first
        result = mongo.db.datasets.delete_one({"_id": ObjectId(dataset_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "Dataset not found"}), 404

        # then file (safely under upload dir)
        if file_path and os.path.exists(file_path):
            upload_dir = os.path.abspath(current_app.config["UPLOAD_FOLDER"])
            file_abs_path = os.path.abspath(file_path)
            if not file_abs_path.startswith(upload_dir):
                current_app.logger.warning(
                    f"Blocked attempt to delete file outside upload directory: {file_path}"
                )
                return jsonify({
                    "message": "Dataset record deleted but file was preserved for security",
                    "warning": "File was outside protected directory"
                }), 200

            os.remove(file_path)
            current_app.logger.info(f"Deleted dataset file: {file_path}")

        return jsonify({"message": "Dataset and file deleted successfully"}), 200

    except Exception as e:
        current_app.logger.exception(f"Delete error: {e}")
        return jsonify({"error": f"Failed to delete dataset: {str(e)}"}), 500


@datasets_bp.route("/api/upload_dataset", methods=["POST"])
@login_required
def upload_dataset():
    """Upload a local dataset (CSV) and register it in MongoDB."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)

    filename = file.filename
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)

    dataset_entry = {
        "filename": filename,
        "filepath": filepath,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Local Upload",
        "user_id": ObjectId(current_user.id),
    }
    mongo.db.datasets.insert_one(dataset_entry)

    return jsonify({"message": "Dataset uploaded successfully"}), 201


@datasets_bp.route("/api/datasets", methods=["GET"])
@login_required
def get_datasets():
    """List datasets uploaded/imported by the current user, plus public datasets."""
    user_id = ObjectId(current_user.id)
    datasets = list(
        mongo.db.datasets.find(
            {"$or": [{"user_id": user_id}, {"is_public": True}]},
            {"_id": 1, "filename": 1, "uploaded_at": 1, "source": 1, "is_public": 1},
        )
    )
    for d in datasets:
        d["_id"] = str(d["_id"])
    return jsonify({"datasets": datasets})