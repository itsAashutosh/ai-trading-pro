import os
import logging
from flask import Flask
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
from backend.config import Config
from backend.extensions import db, migrate, scheduler

def create_app(config_class=Config):
    app = Flask(__name__, template_folder='../templates')
    app.config.from_object(config_class)
    
    # Initialize Extensions
    CORS(app)
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Error Boundaries
    from backend.middleware.error_boundary import register_error_handlers
    register_error_handlers(app)
    
    # Create required directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODELS_FOLDER'], exist_ok=True)
    
    # Configure Logging
    from backend.logging.logger import setup_logger
    setup_logger(app)
    app.logger.info('Backend startup')
    
    # Request ID Middleware
    import uuid
    from flask import request
    
    # Request ID Middleware & Metrics
    import uuid
    import time
    from flask import request, Response
    from backend.metrics.metrics import track_request, track_latency, get_metrics_data
    
    @app.before_request
    def start_timer():
        request.start_time = time.time()
        request.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        
    @app.after_request
    def log_request(response):
        # Calculate duration
        if hasattr(request, 'start_time'):
            duration = time.time() - request.start_time
            track_latency(request.method, request.path, duration)
        
        # Track count
        track_request(request.method, request.path, response.status_code)
        
        if request.path != '/health/live' and request.path != '/metrics':
            app.logger.info(
                f"{request.method} {request.path}",
                extra={
                    'status_code': response.status_code,
                    'content_length': response.content_length,
                    'duration': duration if hasattr(request, 'start_time') else 0,
                    'user_agent': request.user_agent.string
                }
            )
        return response
        
    @app.route('/metrics')
    def metrics():
        data, content_type = get_metrics_data()
        return Response(data, mimetype=content_type)
    
    # Register Blueprints
    from backend.routes.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    from backend.routes.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    from backend.routes.trading import bp as trading_bp
    app.register_blueprint(trading_bp, url_prefix='/api')
    
    from backend.routes.analytics import bp as analytics_bp
    app.register_blueprint(analytics_bp, url_prefix='/api')
    
    from backend.routes.rl import bp as rl_bp
    app.register_blueprint(rl_bp, url_prefix='/api')
    
    from backend.routes.health import bp as health_bp
    app.register_blueprint(health_bp)
    
    # Initialize Scheduler
    if not scheduler.running:
        from backend.services.market_data import update_all_holdings_prices
        scheduler.add_job(
            func=update_all_holdings_prices,
            trigger="interval",
            seconds=60,
            id='price_update_job',
            replace_existing=True,
            args=[app]
        )
        scheduler.start()
    
    return app
