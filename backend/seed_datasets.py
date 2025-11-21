#!/usr/bin/env python3
"""
Seed public datasets into MongoDB.
This script should be run when the backend starts to ensure public datasets are available.
"""
import os
from datetime import datetime
from extensions import mongo

# Public datasets to seed
PUBLIC_DATASETS = [
    {
        "filename": "casehold.csv",
        "relative_path": "casehold/casehold.csv",
        "source": "Public Dataset",
    },
    {
        "filename": "cqa_data.csv",
        "relative_path": "cqa_data.csv",
        "source": "Public Dataset",
    },
    {
        "filename": "deceptive-opinion.csv",
        "relative_path": "deceptive-opinion.csv",
        "source": "Public Dataset",
    },
    {
        "filename": "imdb.csv",
        "relative_path": "imdb.csv",
        "source": "Public Dataset",
    },
    {
        "filename": "bigbenchhard.csv",
        "relative_path": "maveriq/bigbenchhard.csv",
        "source": "Public Dataset",
    },
    {
        "filename": "PubMedQA.csv",
        "relative_path": "qiaojin/PubMedQA.csv",
        "source": "Public Dataset",
    },
]


def seed_public_datasets(upload_folder="uploads"):
    """Seed public datasets into MongoDB if they don't already exist."""
    try:
        uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        seeded_count = 0
        skipped_count = 0
        
        for dataset_info in PUBLIC_DATASETS:
            filename = dataset_info["filename"]
            relative_path = dataset_info["relative_path"]
            filepath = os.path.join(upload_folder, relative_path)
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"Warning: Dataset file not found: {filepath}")
                continue
            
            # Check if dataset already exists (by filename and is_public flag)
            existing = mongo.db.datasets.find_one({
                "filename": filename,
                "is_public": True
            })
            
            if existing:
                print(f"Dataset '{filename}' already exists as public, skipping...")
                skipped_count += 1
                continue
            
            # Insert public dataset
            dataset_entry = {
                "filename": filename,
                "filepath": filepath,
                "uploaded_at": uploaded_at,
                "source": dataset_info["source"],
                "is_public": True,
                # No user_id for public datasets
            }
            
            result = mongo.db.datasets.insert_one(dataset_entry)
            print(f"✓ Seeded public dataset: {filename} (ID: {result.inserted_id})")
            seeded_count += 1
        
        if seeded_count > 0:
            print(f"Dataset seeding completed! Seeded {seeded_count} datasets, skipped {skipped_count} existing.")
        else:
            print(f"All datasets already exist. Skipped {skipped_count} datasets.")
        
    except Exception as e:
        print(f"Error seeding datasets: {e}")
        import traceback
        traceback.print_exc()

