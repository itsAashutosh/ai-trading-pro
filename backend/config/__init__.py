import os
from .runtime import get_env

class Config:
    # Security
    SECRET_KEY = get_env('SECRET_KEY', required=True)
    
    # Database
    SQLALCHEMY_DATABASE_URI = get_env('DATABASE_URL', required=True)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # API Keys
    FINNHUB_API_KEY = get_env('FINNHUB_API_KEY', required=True)
    NEWS_API_KEY = get_env('NEWS_API_KEY', required=True)
    FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'
    
    # Uploads
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MODELS_FOLDER = os.path.join(os.getcwd(), 'models')
    ALLOWED_EXTENSIONS = {'csv'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
