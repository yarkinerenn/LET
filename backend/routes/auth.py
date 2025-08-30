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
_SECRET_KEY = os.getenv("SECRET_KEY")
if not _SECRET_KEY:
    # be loud about it; otherwise decrypt/encrypt will crash later
    raise RuntimeError("SECRET_KEY is missing in environment for Fernet encryption.")
_cipher = Fernet(_SECRET_KEY.encode())

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
        self.grok_api = user_data.get('grok_api', '')
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
    user_data = mongo.db.users.find_one({'_id': ObjectId(current_user.id)}, {'grok_api': 1})
    if user_data and "grok_api" in user_data:
        return decrypt_api_key(user_data['grok_api'])
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
    data = request.json

    if mongo.db.users.find_one({'email': data['email']}):
        return jsonify({"error": "Email already exists"}), 400

    password_hash = generate_password_hash(data['password'])

    openai_api_key   = data.get("openai_api", "")
    grok_api_key     = data.get("grok_api", "")
    deepseek_api_key = data.get("deepseek_api", "")
    openrouter_api_key = data.get("openrouter_api", "")
    gemini_api_key   = data.get("gemini_api", "")

    user_data = {
        'username': data['username'],
        'email': data['email'],
        'password_hash': password_hash,
        'role': 'user',
        'openai_api':   encrypt_api_key(openai_api_key)   if openai_api_key   else "",
        'grok_api':     encrypt_api_key(grok_api_key)     if grok_api_key     else "",
        'deepseek_api': encrypt_api_key(deepseek_api_key) if deepseek_api_key else "",
        'openrouter_api': encrypt_api_key(openrouter_api_key) if openrouter_api_key else "",
        'gemini_api':   encrypt_api_key(gemini_api_key)   if gemini_api_key   else "",
        'preferred_provider':  data.get('preferred_provider', 'openai'),
        'preferred_model':     data.get('preferred_model', 'gpt-3.5-turbo'),
        'preferred_providerex': data.get('preferred_providerex', 'openai'),
        'preferred_modelex':    data.get('preferred_modelex', 'gpt-3.5-turbo')
    }

    result = mongo.db.users.insert_one(user_data)
    return jsonify({"message": "User created successfully", "id": str(result.inserted_id)}), 201

@auth_bp.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user_data = mongo.db.users.find_one({'email': data['email']})
    if not user_data or not check_password_hash(user_data['password_hash'], data['password']):
        return jsonify({"error": "Invalid email or password"}), 401

    login_user(User(user_data))
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

@auth_bp.route('/api/settings/update_api_keys', methods=['POST'])
@login_required
def update_api_keys():
    data = request.json
    openai_api_key    = data.get("openai_api")
    grok_api_key      = data.get("grok_api")
    deepseek_api_key  = data.get("deepseek_api")
    openrouter_api_key = data.get("openrouter_api")
    gemini_api_key    = data.get("gemini_api")

    update_fields = {}
    if openai_api_key:
        update_fields["openai_api"] = encrypt_api_key(openai_api_key)
    if grok_api_key:
        update_fields["grok_api"] = encrypt_api_key(grok_api_key)
    if deepseek_api_key:
        update_fields["deepseek_api"] = encrypt_api_key(deepseek_api_key)
    if openrouter_api_key:
        update_fields["openrouter_api"] = encrypt_api_key(openrouter_api_key)
    if gemini_api_key:
        update_fields["gemini_api"] = encrypt_api_key(gemini_api_key)

    if update_fields:
        mongo.db.users.update_one({"_id": ObjectId(current_user.id)}, {"$set": update_fields})

    return jsonify({"message": "API keys updated successfully"})