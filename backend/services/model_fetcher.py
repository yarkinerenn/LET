"""
Model fetcher service for retrieving available models from various AI providers.
Includes fallback static lists when API calls fail or are unavailable.
"""
import requests
from openai import OpenAI
from groq import Groq
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Static fallback lists (from current frontend implementation)
STATIC_OPENAI_MODELS = [
    {"name": "gpt-5-2025-08-07"},
    {"name": "o4-mini-2025-04-16"},
    {"name": "gpt-4.1-nano-2025-04-14"},
    {"name": "gpt-3.5-turbo"},
    {"name": "gpt-4o-mini-2024-07-18"},
    {"name": "gpt-5-nano-2025-08-07"},
    {"name": "gpt-5-mini-2025-08-07"}
]

STATIC_GROQ_MODELS = [
    {"name": "allam-2-7b"},
    {"name": "llama-3.3-70b-versatile"},
    {"name": "llama-3.1-8b-instant"}
]

STATIC_OPENROUTER_MODELS = [
    {"name": "deepseek/deepseek-r1-0528-qwen3-8b:free"},
    {"name": "deepseek-r1-0528"},
    {"name": "sarvam-m"},
    {"name": "devstral-small"},
    {"name": "gemma-3n-e4b-it"},
    {"name": "google/gemma-3n-e2b-it:free"},
    {"name": "deephermes-3-mistral-24b-preview"},
    {"name": "phi-4-reasoning-plus"},
    {"name": "phi-4-reasoning"},
    {"name": "internvl3-14b"},
    {"name": "internvl3-2b"},
    {"name": "deepseek-prover-v2"},
    {"name": "qwen3-30b-a3b"},
    {"name": "qwen3-8b"},
    {"name": "qwen3-14b"},
    {"name": "qwen3-32b"},
    {"name": "qwen3-235b-a22b"},
    {"name": "deepseek-r1t-chimera"},
    {"name": "mai-ds-r1"},
    {"name": "glm-z1-32b"},
    {"name": "glm-4-32b"},
    {"name": "shisa-v2-llama3.3-70b"},
    {"name": "qwq-32b-arliai-rpr-v1"},
    {"name": "deepcoder-14b-preview"},
    {"name": "kimi-vl-a3b-thinking"},
    {"name": "llama-3.3-nemotron-super-49b-v1"},
    {"name": "llama-3.1-nemotron-ultra-253b-v1"},
    {"name": "llama-4-maverick"},
    {"name": "llama-4-scout"},
    {"name": "deepseek-v3-base"},
    {"name": "qwen2.5-vl-3b-instruct"},
    {"name": "gemini-2.5-pro-exp-03-25"},
    {"name": "qwen2.5-vl-32b-instruct"},
    {"name": "deepseek-chat-v3-0324"},
    {"name": "qwerky-72b"},
    {"name": "mistral-small-3.1-24b-instruct"},
    {"name": "olympiccoder-32b"},
    {"name": "gemma-3-1b-it"},
    {"name": "gemma-3-4b-it"},
    {"name": "gemma-3-12b-it"},
    {"name": "reka-flash-3"},
    {"name": "gemma-3-27b-it"},
    {"name": "deepseek-r1-zero"},
    {"name": "qwq-32b"},
    {"name": "moonlight-16b-a3b-instruct"},
    {"name": "deephermes-3-llama-3-8b-preview"},
    {"name": "dolphin3.0-r1-mistral-24b"},
    {"name": "dolphin3.0-mistral-24b"},
    {"name": "qwen2.5-vl-72b-instruct"},
    {"name": "mistral-small-24b-instruct-2501"},
    {"name": "deepseek-r1-distill-qwen-32b"},
    {"name": "deepseek-r1-distill-qwen-14b"},
    {"name": "deepseek-r1-distill-llama-70b"},
    {"name": "deepseek-r1"},
    {"name": "deepseek-chat"},
    {"name": "gemini-2.0-flash-exp"},
    {"name": "llama-3.3-70b-instruct"},
    {"name": "qwen-2.5-coder-32b-instruct"},
    {"name": "qwen-2.5-7b-instruct"},
    {"name": "llama-3.2-3b-instruct"},
    {"name": "llama-3.2-1b-instruct"},
    {"name": "llama-3.2-11b-vision-instruct"},
    {"name": "qwen-2.5-72b-instruct"},
    {"name": "qwen-2.5-vl-7b-instruct"},
    {"name": "llama-3.1-405b"},
    {"name": "llama-3.1-8b-instruct"},
    {"name": "mistral-nemo"},
    {"name": "gemma-2-9b-it"},
    {"name": "mistral-7b-instruct"}
]

STATIC_GEMINI_MODELS = [
    {"name": "models/gemini-1.5-flash-8b"},
    {"name": "gemini-1.5-flash"},
    {"name": "gemini-2.0-flash-exp"},
    {"name": "gemini-2.5-pro-exp-03-25"}
]

STATIC_DEEPSEEK_MODELS = [
    {"name": "deepseek-chat"},
    {"name": "deepseek-coder"},
    {"name": "deepseek-chat-v3-0324"}
]

STATIC_OLLAMA_MODELS = [
    {"name": "jsk/bio-mistral"},
    {"name": "phi3.5:latest"},
    {"name": "gemma:2b"},
    {"name": "llama3.1:8b"},
    {"name": "mistral:7b"}
]


def fetch_openai_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch available models from OpenAI API."""
    if not api_key:
        print("⚠️  OpenAI: No API key provided, using static fallback models")
        logger.warning("No OpenAI API key provided, using static fallback")
        return STATIC_OPENAI_MODELS
    
    try:
        print("🔄 OpenAI: Fetching models from API...")
        client = OpenAI(api_key=api_key)
        response = client.models.list()
        models = [{"name": model.id} for model in response.data if model.id.startswith(('gpt-', 'o1-', 'o4-'))]
        if models:
            print(f"✅ OpenAI: Successfully fetched {len(models)} models from API")
            logger.info(f"Successfully fetched {len(models)} OpenAI models from API")
        else:
            print("⚠️  OpenAI: API returned no models, using static fallback")
        return models if models else STATIC_OPENAI_MODELS
    except Exception as e:
        print(f"❌ OpenAI: Error fetching models from API: {e}")
        print("⚠️  OpenAI: Falling back to static models")
        logger.error(f"Error fetching OpenAI models: {e}")
        return STATIC_OPENAI_MODELS


def fetch_groq_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch available models from Groq API."""
    if not api_key:
        print("⚠️  Groq: No API key provided, using static fallback models")
        logger.warning("No Groq API key provided, using static fallback")
        return STATIC_GROQ_MODELS
    
    try:
        # Groq doesn't have a direct models.list() endpoint, so we'll use static list
        # If Groq adds this endpoint in the future, we can update this
        print("ℹ️  Groq: Using static models list (API endpoint not available)")
        logger.info("Using static Groq models list (API endpoint not available)")
        return STATIC_GROQ_MODELS
    except Exception as e:
        print(f"❌ Groq: Error fetching models: {e}")
        print("⚠️  Groq: Falling back to static models")
        logger.error(f"Error fetching Groq models: {e}")
        return STATIC_GROQ_MODELS


def fetch_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch available models from OpenRouter API."""
    if not api_key:
        print("⚠️  OpenRouter: No API key provided, using static fallback models")
        logger.warning("No OpenRouter API key provided, using static fallback")
        return STATIC_OPENROUTER_MODELS
    
    try:
        print("🔄 OpenRouter: Fetching models from API...")
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        response = client.models.list()
        models = [{"name": model.id} for model in response.data]
        if models:
            print(f"✅ OpenRouter: Successfully fetched {len(models)} models from API")
            logger.info(f"Successfully fetched {len(models)} OpenRouter models from API")
        else:
            print("⚠️  OpenRouter: API returned no models, using static fallback")
        return models if models else STATIC_OPENROUTER_MODELS
    except Exception as e:
        print(f"❌ OpenRouter: Error fetching models from API: {e}")
        print("⚠️  OpenRouter: Falling back to static models")
        logger.error(f"Error fetching OpenRouter models: {e}")
        return STATIC_OPENROUTER_MODELS


def fetch_gemini_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch available models from Gemini API."""
    if not api_key:
        print("⚠️  Gemini: No API key provided, using static fallback models")
        logger.warning("No Gemini API key provided, using static fallback")
        return STATIC_GEMINI_MODELS
    
    try:
        print("🔄 Gemini: Fetching models from API...")
        # Gemini uses OpenAI-compatible endpoint
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key
        )
        response = client.models.list()
        models = [{"name": model.id} for model in response.data]
        if models:
            print(f"✅ Gemini: Successfully fetched {len(models)} models from API")
            logger.info(f"Successfully fetched {len(models)} Gemini models from API")
        else:
            print("⚠️  Gemini: API returned no models, using static fallback")
        return models if models else STATIC_GEMINI_MODELS
    except Exception as e:
        print(f"❌ Gemini: Error fetching models from API: {e}")
        print("⚠️  Gemini: Falling back to static models")
        logger.error(f"Error fetching Gemini models: {e}")
        return STATIC_GEMINI_MODELS


def fetch_deepseek_models(api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """Fetch available models from DeepSeek API."""
    if not api_key:
        print("⚠️  DeepSeek: No API key provided, using static fallback models")
        logger.warning("No DeepSeek API key provided, using static fallback")
        return STATIC_DEEPSEEK_MODELS
    
    try:
        print("🔄 DeepSeek: Fetching models from API...")
        client = OpenAI(base_url="https://api.deepseek.com/v1", api_key=api_key)
        response = client.models.list()
        models = [{"name": model.id} for model in response.data]
        if models:
            print(f"✅ DeepSeek: Successfully fetched {len(models)} models from API")
            logger.info(f"Successfully fetched {len(models)} DeepSeek models from API")
        else:
            print("⚠️  DeepSeek: API returned no models, using static fallback")
        return models if models else STATIC_DEEPSEEK_MODELS
    except Exception as e:
        print(f"❌ DeepSeek: Error fetching models from API: {e}")
        print("⚠️  DeepSeek: Falling back to static models")
        logger.error(f"Error fetching DeepSeek models: {e}")
        return STATIC_DEEPSEEK_MODELS


def fetch_ollama_models() -> List[Dict[str, str]]:
    """Fetch available models from local Ollama API."""
    try:
        print("🔄 Ollama: Fetching models from local API...")
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [{"name": model["name"]} for model in data.get("models", [])]
            if models:
                print(f"✅ Ollama: Successfully fetched {len(models)} models from local API")
                logger.info(f"Successfully fetched {len(models)} Ollama models from local API")
            else:
                print("⚠️  Ollama: Local API returned no models, using static fallback")
            return models if models else STATIC_OLLAMA_MODELS
        else:
            print("⚠️  Ollama: Local API not available (status code: {}), using static fallback".format(response.status_code))
            logger.warning("Ollama API not available, using static fallback")
            return STATIC_OLLAMA_MODELS
    except Exception as e:
        print(f"⚠️  Ollama: Local API not available ({e}), using static fallback")
        logger.warning(f"Ollama API not available: {e}, using static fallback")
        return STATIC_OLLAMA_MODELS


def get_static_models() -> Dict[str, List[Dict[str, str]]]:
    """Get all static fallback models."""
    return {
        "openai": STATIC_OPENAI_MODELS,
        "groq": STATIC_GROQ_MODELS,
        "openrouter": STATIC_OPENROUTER_MODELS,
        "gemini": STATIC_GEMINI_MODELS,
        "deepseek": STATIC_DEEPSEEK_MODELS,
        "ollama": STATIC_OLLAMA_MODELS
    }

