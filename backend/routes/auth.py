# auth.py
from flask import Blueprint, jsonify, request
from flask_login import (
    UserMixin, login_user, logout_user, current_user, login_required
)
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId

# extensions
from extensions import mongo, login_manager

# ---------------- Crypto helpers (moved from app.py) ----------------
# If you already extracted these into another module, import from there instead.
import os
from dotenv import load_dotenv
from cryptography.fernet import Fernet

load_dotenv()
_SECRET_KEY = os.getenv("SECRET_KEY") or os.getenv("FLASK_SECRET_KEY")
if not _SECRET_KEY:
    # be loud about it; otherwise decrypt/encrypt will crash later
    raise RuntimeError("SECRET_KEY or FLASK_SECRET_KEY is missing in environment for Fernet encryption.")

# Ensure SECRET_KEY is a valid Fernet key (32 bytes base64-encoded)
# If it's not valid, generate a key from it
try:
    _cipher = Fernet(_SECRET_KEY.encode())
except ValueError:
    # If SECRET_KEY is not a valid Fernet key, derive one from it
    import hashlib
    import base64
    # Use SHA256 to hash the secret key to get 32 bytes, then base64 encode
    key_bytes = hashlib.sha256(_SECRET_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    _cipher = Fernet(fernet_key)

def encrypt_api_key(key_str: str) -> str:
    return _cipher.encrypt(key_str.encode()).decode()

def decrypt_api_key(token_str: str) -> str:
    return _cipher.decrypt(token_str.encode()).decode()

# ---------------- Blueprint ----------------
auth_bp = Blueprint("auth", __name__)

# ---------------- User model + loader ----------------
class User(UserMixin):
    """User class"""
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']
        self.role = user_data.get('role', 'user')
        self.openai_api = user_data.get('openai_api', '')
        self.groq_api = user_data.get('groq_api', '')
        self.deepseek_api = user_data.get('deepseek_api', '')
        self.gemini_api = user_data.get('gemini_api', '')
        self.preferred_provider = user_data.get('preferred_provider', 'openai')
        self.preferred_model = user_data.get('preferred_model', 'gpt-3.5-turbo')
        self.preferred_providerex = user_data.get('preferred_providerex', 'openai')
        self.preferred_modelex = user_data.get('preferred_modelex', 'gpt-3.5-turbo')

    @staticmethod
    def get(user_id):
        user_data = mongo.db.users.find_one({'_id': ObjectId(user_id)})
        if not user_data:
            return None
        return User(user_data)

    @staticmethod
    def find_by_email(email):
        user_data = mongo.db.users.find_one({'email': email})
        if not user_data:
            return None
        return User(user_data)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# ---------------- API key getters ----------------
def get_user_api_key_gemini():
    if not current_user.is_authenticated:
        return None
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'gemini_api': 1})
    if user_data and "gemini_api" in user_data:
        return decrypt_api_key(user_data['gemini_api'])
    return None

def get_user_api_key_openai():
    if not current_user.is_authenticated:
        return None
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'openai_api': 1})
    if user_data and "openai_api" in user_data:
        return decrypt_api_key(user_data['openai_api'])
    return None

def get_user_api_key_deepseek_api():
    if not current_user.is_authenticated:
        return None
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'deepseek_api': 1})
    if user_data and "deepseek_api" in user_data:
        return decrypt_api_key(user_data['deepseek_api'])
    return None

def get_user_api_key_groq():
    if not current_user.is_authenticated:
        return None
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'groq_api': 1})
    if user_data and "groq_api" in user_data:
        return decrypt_api_key(user_data['groq_api'])
    return None

def get_user_api_key_openrouter():
    if not current_user.is_authenticated:
        return None
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'openrouter_api': 1})
    if user_data and "openrouter_api" in user_data:
        return decrypt_api_key(user_data['openrouter_api'])
    return None

# ---------------- Auth & settings routes (converted to blueprint) ----------------
@auth_bp.route('/api/register', methods=['POST'])
def register():
    try:
        print(f"Registration request received: {request.method}")
        print(f"Request origin: {request.headers.get('Origin')}")
        print(f"Request headers: {dict(request.headers)}")
        data = request.json
        print(f"Request data keys: {list(data.keys()) if data else 'None'}")
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate required fields
        if not data.get('email'):
            return jsonify({"error": "Email is required"}), 400
        if not data.get('password'):
            return jsonify({"error": "Password is required"}), 400

        # Check if email already exists
        if mongo.db.users.find_one({'email': data['email']}):
            return jsonify({"error": "Email already exists"}), 400

        password_hash = generate_password_hash(data['password'])

        # Auto-generate username from email if not provided
        username = data.get('username')
        if not username:
            # Use the part before @ as username, or use email if no @ found
            username = data['email'].split('@')[0] if '@' in data['email'] else data['email']

        openai_api_key   = data.get("openai_api", "")
        groq_api_key     = data.get("groq_api", "")
        deepseek_api_key = data.get("deepseek_api", "")
        openrouter_api_key = data.get("openrouter_api", "")
        gemini_api_key   = data.get("gemini_api", "")

        # Helper function to safely encrypt API keys
        def safe_encrypt(key_str):
            if key_str and key_str.strip():
                try:
                    return encrypt_api_key(key_str)
                except Exception as e:
                    print(f"Warning: Failed to encrypt API key: {e}")
                    return ""  # Return empty string if encryption fails
            return ""

        user_data = {
            'username': username,
            'email': data['email'],
            'password_hash': password_hash,
            'role': 'user',
            'openai_api':   safe_encrypt(openai_api_key),
            'groq_api':     safe_encrypt(groq_api_key),
            'deepseek_api': safe_encrypt(deepseek_api_key),
            'openrouter_api': safe_encrypt(openrouter_api_key),
            'gemini_api':   safe_encrypt(gemini_api_key),
            'preferred_provider':  data.get('preferred_provider', 'openai'),
            'preferred_model':     data.get('preferred_model', 'gpt-3.5-turbo'),
            'preferred_providerex': data.get('preferred_providerex', 'openai'),
            'preferred_modelex':    data.get('preferred_modelex', 'gpt-3.5-turbo')
        }

        result = mongo.db.users.insert_one(user_data)
        return jsonify({"message": "User created successfully", "id": str(result.inserted_id)}), 201

    except Exception as e:
        import traceback
        print(f"Registration error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@auth_bp.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user_data = mongo.db.users.find_one({'email': data['email']})
    if not user_data or not check_password_hash(user_data['password_hash'], data['password']):
        return jsonify({"error": "Invalid email or password"}), 401

    remember_me = data.get('remember_me', False)
    login_user(User(user_data), remember=remember_me)
    return jsonify({
        "message": "Logged in successfully",
        "user": {"id": str(user_data['_id']), "username": user_data['username']}
    })

@auth_bp.route('/api/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

@auth_bp.route('/api/check_auth', methods=['GET'])
def check_auth():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "user": {"id": current_user.id, "username": current_user.username}
        })
    return jsonify({"authenticated": False})

@auth_bp.route('/api/settings/update_preferred_classification', methods=['POST'])
@login_required
def update_preferred_classification():
    data = request.json
    preferred_provider = data.get('preferred_provider', 'openai')
    preferred_model = data.get('preferred_model', 'gpt-3.5-turbo') or 'gpt-3.5-turbo'
    mongo.db.users.update_one(
        {'_id': ObjectId(current_user.id)},
        {'$set': {'preferred_provider': preferred_provider, 'preferred_model': preferred_model}}
    )
    return jsonify({"message": "Classification preferences updated successfully"}), 200

@auth_bp.route('/api/settings/update_preferred_explanation', methods=['POST'])
@login_required
def update_preferred_explanation():
    data = request.json
    preferred_providerex = data.get('preferred_providerex', 'openai')
    preferred_modelex = data.get('preferred_modelex', 'gpt-3.5-turbo')
    mongo.db.users.update_one(
        {'_id': ObjectId(current_user.id)},
        {'$set': {'preferred_providerex': preferred_providerex, 'preferred_modelex': preferred_modelex}}
    )
    return jsonify({"message": "Explanation preferences updated successfully"}), 200

@auth_bp.route('/api/settings/get_preferences', methods=['GET'])
@login_required
def get_preferences():
    """Get user preferences for classification and explanation"""
    user_data = mongo.db.users.find_one(
        {'_id': ObjectId(current_user.id)},
        {'preferred_provider': 1, 'preferred_model': 1, 'preferred_providerex': 1, 'preferred_modelex': 1}
    )
    
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "preferred_provider": user_data.get('preferred_provider', 'openai'),
        "preferred_model": user_data.get('preferred_model', 'gpt-3.5-turbo'),
        "preferred_providerex": user_data.get('preferred_providerex', 'openai'),
        "preferred_modelex": user_data.get('preferred_modelex', 'gpt-3.5-turbo')
    }), 200

@auth_bp.route('/api/settings/get_api_keys_status', methods=['GET'])
@login_required
def get_api_keys_status():
    """Get status of which API keys are stored (without decrypting them)"""
    user_data = mongo.db.users.find_one(
        {'_id': ObjectId(current_user.id)},
        {'openai_api': 1, 'groq_api': 1, 'deepseek_api': 1, 'openrouter_api': 1, 'gemini_api': 1}
    )
    
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "openai_api": bool(user_data.get('openai_api')),
        "groq_api": bool(user_data.get('groq_api')),
        "deepseek_api": bool(user_data.get('deepseek_api')),
        "openrouter_api": bool(user_data.get('openrouter_api')),
        "gemini_api": bool(user_data.get('gemini_api'))
    }), 200

@auth_bp.route('/api/settings/update_api_keys', methods=['POST'])
@login_required
def update_api_keys():
    data = request.json
    openai_api_key    = data.get("openai_api")
    groq_api_key      = data.get("groq_api")
    deepseek_api_key  = data.get("deepseek_api")
    openrouter_api_key = data.get("openrouter_api")
    gemini_api_key    = data.get("gemini_api")

    update_fields = {}
    if openai_api_key:
        update_fields["openai_api"] = encrypt_api_key(openai_api_key)
    if groq_api_key:
        update_fields["groq_api"] = encrypt_api_key(groq_api_key)
    if deepseek_api_key:
        update_fields["deepseek_api"] = encrypt_api_key(deepseek_api_key)
    if openrouter_api_key:
        update_fields["openrouter_api"] = encrypt_api_key(openrouter_api_key)
    if gemini_api_key:
        update_fields["gemini_api"] = encrypt_api_key(gemini_api_key)

    if update_fields:
        mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": update_fields})

    return jsonify({"message": "API keys updated successfully"})

@auth_bp.route('/api/settings/delete_api_key', methods=['POST'])
@login_required
def delete_api_key():
    """Delete a specific API key from user profile"""
    data = request.json
    key_type = data.get("key_type")  # e.g., "openai_api", "groq_api", etc.
    
    if not key_type:
        return jsonify({"error": "key_type is required"}), 400
    
    valid_key_types = ['openai_api', 'groq_api', 'deepseek_api', 'openrouter_api', 'gemini_api']
    if key_type not in valid_key_types:
        return jsonify({"error": f"Invalid key_type. Must be one of: {', '.join(valid_key_types)}"}), 400
    
    try:
        # Unset (delete) the API key field
        mongo.db.users.update_one(
            {"_id": ObjectId(current_user.id)},
            {"$unset": {key_type: ""}}
        )
        
        return jsonify({"message": f"{key_type} deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to delete API key: {str(e)}"}), 500