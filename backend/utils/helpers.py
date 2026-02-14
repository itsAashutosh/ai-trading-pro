import pandas as pd
from flask import current_app

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def validate_csv_data(filepath):
    """Validate CSV file structure"""
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        if 'Close' not in df.columns:
            return False, "CSV must contain a 'Close' column"
        
        # Validate Close column
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().all():
            return False, "Close column must contain numeric values"
        
        # Check for sufficient data
        if len(df) < 10:
            return False, "CSV must contain at least 10 rows of data"
        
        return True, "Valid CSV file"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"
