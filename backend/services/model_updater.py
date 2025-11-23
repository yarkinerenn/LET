"""
Model updater service for scheduled updates of available models from providers.
Updates MongoDB cache with fresh model lists.
"""
from datetime import datetime, timedelta
from typing import Dict, List
from bson import ObjectId
import logging

from extensions import mongo
from services.model_fetcher import (
    fetch_openai_models,
    fetch_groq_models,
    fetch_openrouter_models,
    fetch_gemini_models,
    fetch_deepseek_models,
    fetch_ollama_models,
    get_static_models
)

logger = logging.getLogger(__name__)

# Cache expiration time (24 hours)
CACHE_EXPIRY_HOURS = 24


def get_cached_models() -> Dict[str, List[Dict[str, str]]]:
    """Get models from cache if available and fresh."""
    try:
        cache_docs = mongo.db.model_cache.find({})
        cached_models = {}
        
        for doc in cache_docs:
            provider = doc.get("provider")
            last_updated = doc.get("last_updated")
            
            # Check if cache is still fresh
            if last_updated:
                if isinstance(last_updated, str):
                    last_updated = datetime.fromisoformat(last_updated)
                age = datetime.now() - last_updated
                if age < timedelta(hours=CACHE_EXPIRY_HOURS):
                    cached_models[provider] = doc.get("models", [])
        
        return cached_models
    except Exception as e:
        logger.error(f"Error reading cached models: {e}")
        return {}


def update_all_models(use_api_keys: bool = True) -> Dict[str, List[Dict[str, str]]]:
    """
    Update all model lists from APIs and cache them.
    
    Args:
        use_api_keys: If True, attempts to use API keys from a sample user to fetch models.
                     If False, uses static fallback lists.
    
    Returns:
        Dictionary mapping provider names to their model lists
    """
    print("\n" + "="*60)
    print("🔄 Starting model list update...")
    print("="*60)
    all_models = {}
    
    try:
        # Try to fetch from APIs if we have API keys
        if use_api_keys:
            print("📋 Attempting to fetch models from provider APIs...")
            # Get API keys from any user (we'll use the first user we find)
            # In production, you might want to use a system account or admin account
            sample_user = mongo.db.users.find_one({})
            
            if sample_user:
                openai_key = None
                groq_key = None
                openrouter_key = None
                gemini_key = None
                deepseek_key = None
                
                # Decrypt API keys if available
                # Import here to avoid circular imports
                import os
                from dotenv import load_dotenv
                from cryptography.fernet import Fernet
                
                load_dotenv()
                secret_key = os.getenv("SECRET_KEY")
                if secret_key:
                    cipher = Fernet(secret_key.encode())
                    
                    def decrypt_api_key(key_str: str) -> str:
                        return cipher.decrypt(key_str.encode()).decode()
                    
                    if sample_user.get("openai_api"):
                        try:
                            openai_key = decrypt_api_key(sample_user["openai_api"])
                        except:
                            pass
                    if sample_user.get("grok_api"):
                        try:
                            groq_key = decrypt_api_key(sample_user["grok_api"])
                        except:
                            pass
                    if sample_user.get("openrouter_api"):
                        try:
                            openrouter_key = decrypt_api_key(sample_user["openrouter_api"])
                        except:
                            pass
                    if sample_user.get("gemini_api"):
                        try:
                            gemini_key = decrypt_api_key(sample_user["gemini_api"])
                        except:
                            pass
                    if sample_user.get("deepseek_api"):
                        try:
                            deepseek_key = decrypt_api_key(sample_user["deepseek_api"])
                        except:
                            pass
                
                # Fetch models from APIs
                all_models["openai"] = fetch_openai_models(openai_key)
                all_models["groq"] = fetch_groq_models(groq_key)
                all_models["openrouter"] = fetch_openrouter_models(openrouter_key)
                all_models["gemini"] = fetch_gemini_models(gemini_key)
                all_models["deepseek"] = fetch_deepseek_models(deepseek_key)
            else:
                # No users found, use static lists
                all_models = get_static_models()
        else:
            # Use static lists
            all_models = get_static_models()
        
        # Ollama doesn't require API keys
        all_models["ollama"] = fetch_ollama_models()
        
        # Update cache
        print("\n💾 Updating model cache in database...")
        now = datetime.now()
        for provider, models in all_models.items():
            mongo.db.model_cache.update_one(
                {"provider": provider},
                {
                    "$set": {
                        "provider": provider,
                        "models": models,
                        "last_updated": now,
                        "source": "api" if use_api_keys and provider != "ollama" else "static"
                    }
                },
                upsert=True
            )
            print(f"  ✓ Cached {len(models)} models for {provider}")
        
        print("\n" + "="*60)
        print(f"✅ Model list update completed successfully!")
        print(f"   Updated {len(all_models)} providers")
        print("="*60 + "\n")
        logger.info(f"Successfully updated model cache for {len(all_models)} providers")
        return all_models
        
    except Exception as e:
        print(f"\n❌ Error updating models: {e}")
        print("⚠️  Attempting to use cached models...")
        logger.error(f"Error updating models: {e}")
        # Return cached models if update fails
        cached = get_cached_models()
        if cached:
            print(f"✅ Using cached models for {len(cached)} providers\n")
            return cached
        # Fallback to static lists
        print("⚠️  No cache available, using static fallback models\n")
        return get_static_models()


def get_all_models(force_refresh: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """
    Get all models, using cache if available and fresh, otherwise fetching fresh data.
    
    Args:
        force_refresh: If True, bypasses cache and fetches fresh data
    
    Returns:
        Dictionary mapping provider names to their model lists
    """
    if force_refresh:
        return update_all_models()
    
    # Check cache first
    cached = get_cached_models()
    
    # If we have cached data for all providers, return it
    static_models = get_static_models()
    if len(cached) >= len(static_models):
        return cached
    
    # Otherwise, update and return fresh data
    return update_all_models()

