from flask import jsonify, current_app
from werkzeug.exceptions import HTTPException
import traceback

def register_error_handlers(app):
    @app.errorhandler(Exception)
    def handle_exception(e):
        # Pass through HTTP errors
        if isinstance(e, HTTPException):
            return jsonify({
                'error': e.name,
                'message': e.description,
                'code': e.code
            }), e.code
            
        # Log unexpected errors
        current_app.logger.error(
            f"Unhandled Exception: {str(e)}",
            extra={'stack_trace': traceback.format_exc()}
        )
        
        # Return generic 500
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. Please contact support.',
            'code': 500
        }), 500
