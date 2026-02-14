from flask import Blueprint, jsonify, current_app
from backend.extensions import db, scheduler
from sqlalchemy import text

bp = Blueprint('health', __name__, url_prefix='/health')

@bp.route('/live', methods=['GET'])
def liveness_probe():
    """
    K8s Liveness Probe.
    Returns 200 if the app is running.
    """
    return jsonify({'status': 'alive'}), 200

@bp.route('/ready', methods=['GET'])
def readiness_probe():
    """
    K8s Readiness Probe.
    Returns 200 if all dependencies are healthy.
    """
    dependency_status = {
        'database': False,
        'scheduler': False
    }
    
    # Check Database
    try:
        db.session.execute(text('SELECT 1'))
        dependency_status['database'] = True
    except Exception as e:
        current_app.logger.error(f"Readiness check failed (DB): {str(e)}")
        
    # Check Scheduler
    if scheduler.running:
        dependency_status['scheduler'] = True
    else:
        current_app.logger.error("Readiness check failed (Scheduler not running)")
        
    # Overall Status
    if all(dependency_status.values()):
        return jsonify({
            'status': 'ready',
            'dependencies': dependency_status
        }), 200
    else:
        return jsonify({
            'status': 'not_ready',
            'dependencies': dependency_status
        }), 503
