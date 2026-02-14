import logging
import json
import uuid
from flask import request, has_request_context

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'funcName': record.funcName,
            'timestamp': self.formatTime(record, self.datefmt),
        }
        
        # Add request context if available
        if has_request_context():
            log_obj['request_id'] = getattr(request, 'request_id', 'unknown')
            log_obj['method'] = request.method
            log_obj['path'] = request.path
            log_obj['ip'] = request.remote_addr
            
        # Add extra fields passed in extra={}
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)
            
        return json.dumps(log_obj)

def setup_logger(app):
    # Remove default handlers
    app.logger.handlers.clear()
    
    # Create handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    # Configure root logger and app logger
    app.logger.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    
    # Also capture Werkzeug logs but keep them simple/JSON?
    # For now, let's just make sure app logs are JSON.
    logging.getLogger('werkzeug').setLevel(logging.WARNING) # Reduce noise
    
    return app.logger
