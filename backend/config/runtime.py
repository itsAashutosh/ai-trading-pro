import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env(key, default=None, required=False):
    """
    Get environment variable with validation.
    
    Args:
        key (str): Environment variable name
        default (any): Default value if not found
        required (bool): If True, raise error if missing
        
    Returns:
        str: Environment variable value
    """
    value = os.environ.get(key, default)
    
    if required and not value:
        raise ValueError(f"CRITICAL: Missing required environment variable '{key}'. Check your .env file.")
        
    return value
