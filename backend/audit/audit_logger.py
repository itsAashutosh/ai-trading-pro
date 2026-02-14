import logging
import json
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Configure Audit Logger
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False  # Don't propagate to root logger

if not os.path.exists('logs'):
    os.makedirs('logs')

handler = RotatingFileHandler('logs/audit.log', maxBytes=10*1024*1024, backupCount=100)
handler.setFormatter(logging.Formatter('%(message)s'))
audit_logger.addHandler(handler)

def log_audit_event(event_type, user_id=None, **details):
    """
    Log an immutable audit event.
    
    Args:
        event_type (str): Type of event (e.g., 'TRADE_EXECUTED', 'TRAINING_STARTED')
        user_id (int, optional): User ID associated with the event
        details (dict): Additional details to log
    """
    event = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'event_type': event_type.upper(),
        'user_id': user_id,
        'details': details
    }
    
    # Write JSON line
    audit_logger.info(json.dumps(event))
