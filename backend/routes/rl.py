import os
import json
import threading
import pickle
import numpy as np
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app, send_from_directory, Response
from werkzeug.utils import secure_filename

from backend.extensions import db
from backend.models.rl import RLTrainingJob, RLAgentState
from backend.tasks.rl_worker import run_rl_training
from backend.utils.helpers import allowed_file, validate_csv_data

bp = Blueprint('rl', __name__)

# ==================== RL TRAINING ROUTES ====================

@bp.route('/rl-training/upload', methods=['POST'])
def upload_training_data():
    """Upload CSV file for RL training"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Validate CSV
        is_valid, message = validate_csv_data(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({'error': message}), 400
        
        # Read data preview
        import pandas as pd
        df = pd.read_csv(filepath)
        preview = {
            'rows': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': df['Date'].iloc[0] if 'Date' in df.columns else None,
                'end': df['Date'].iloc[-1] if 'Date' in df.columns else None
            },
            'close_stats': {
                'min': float(df['Close'].min()),
                'max': float(df['Close'].max()),
                'mean': float(df['Close'].mean())
            }
        }
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': unique_filename,
            'filepath': filepath,
            'preview': preview
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@bp.route('/rl-training/start', methods=['POST'])
def start_training():
    """Start RL agent training"""
    data = request.json
    
    # Validate required fields
    required_fields = ['user_id', 'filename', 'episodes', 'learning_rate', 
                      'discount_factor', 'epsilon', 'epsilon_decay', 'initial_balance']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Check if file exists
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], data['filename'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    # Check Resource Limits
    from backend.guards.training_limits import TrainingLimits
    passed, reason = TrainingLimits.check_training_request(filepath, data)
    if not passed:
        return jsonify({'error': f"Resource Limit: {reason}"}), 400
    
    try:
        # Create training job
        job = RLTrainingJob(
            user_id=data['user_id'],
            filename=data['filename'],
            episodes=int(data['episodes']),
            learning_rate=float(data['learning_rate']),
            discount_factor=float(data['discount_factor']),
            epsilon=float(data['epsilon']),
            epsilon_decay=float(data['epsilon_decay']),
            initial_balance=float(data['initial_balance']),
            status='pending'
        )
        
        db.session.add(job)
        db.session.commit()
        
        # AUDIT LOG
        from backend.audit.audit_logger import log_audit_event
        log_audit_event('TRAINING_STARTED', user_id=data['user_id'], job_id=job.id, params=data)
        
        # Start training in background thread
        params = {
            'episodes': job.episodes,
            'learning_rate': job.learning_rate,
            'discount_factor': job.discount_factor,
            'epsilon': job.epsilon,
            'epsilon_decay': job.epsilon_decay,
            'initial_balance': job.initial_balance
        }
        
        # Pass the actual app object to the thread
        # current_app is a proxy, we need the real app object
        app_obj = current_app._get_current_object()
        
        thread = threading.Thread(
            target=run_rl_training,
            args=(app_obj, job.id, filepath, params)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Training started',
            'job': job.to_dict()
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500


@bp.route('/rl-training/jobs/<int:user_id>', methods=['GET'])
def get_training_jobs(user_id):
    """Get all training jobs for a user"""
    jobs = RLTrainingJob.query.filter_by(user_id=user_id).order_by(
        RLTrainingJob.created_at.desc()
    ).all()
    return jsonify([job.to_dict() for job in jobs])


@bp.route('/rl-training/job/<int:job_id>', methods=['GET'])
def get_training_job(job_id):
    """Get specific training job status"""
    job = RLTrainingJob.query.get_or_404(job_id)
    return jsonify(job.to_dict())


@bp.route("/rl-training/job/<int:jobid>/results", methods=["GET"])
def get_training_results(jobid):
    """Fetches the training results from the job-specific JSON file."""
    job = db.session.get(RLTrainingJob, jobid)
    if not job:
        return jsonify({"error": f"Job {jobid} not found"}), 404

    if job.status != "completed":
        return jsonify({"error": "Training not completed or file not ready"}), 400
    
    # Check "results" directory relative to current working directory
    # Assuming run.py is executed from root
    results_path = os.path.join(
        os.getcwd(), 
        "results", 
        f"results_{jobid}.json"
    )
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({"error": "Results file not found on disk. Training likely failed to save data."}), 404


@bp.route('/rl-training/job/<int:job_id>/plot', methods=['GET'])
def get_training_plot(job_id):
    """Get training performance plot"""
    job = RLTrainingJob.query.get_or_404(job_id)
    
    if job.status != 'completed':
        return jsonify({'error': 'Training not completed'}), 400
    
    # Determine plot path (assuming single plot per job overwritten or unique?)
    # Original code used 'trading_plot.png' always.
    # rl_worker.py saved it as 'trading_plot.png' in results dir.
    # This means concurrent jobs might overwrite each other's plot if we don't name them uniquely.
    # But sticking to original logic for now:
    plot_path = os.path.join(os.getcwd(), 'results', 'trading_plot.png')
    
    if os.path.exists(plot_path):
        return send_from_directory(
            os.path.dirname(plot_path),
            'trading_plot.png',
            mimetype='image/png'
        )
    else:
        return jsonify({'error': 'Plot file not found'}), 404


@bp.route('/rl-training/job/<int:job_id>', methods=['DELETE'])
def delete_training_job(job_id):
    """Delete a training job"""
    job = RLTrainingJob.query.get_or_404(job_id)
    
    # Delete uploaded file if exists
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], job.filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass
    
    db.session.delete(job)
    db.session.commit()
    
    return jsonify({'message': 'Training job deleted'})


@bp.route('/rl-training/job/<int:job_id>/download', methods=['GET'])
def download_training_results(job_id):
    """Download training results as JSON file"""
    job = db.session.get(RLTrainingJob, job_id)
    if not job:
        return jsonify({"error": f"Job {job_id} not found"}), 404

    if job.status != "completed":
        return jsonify({"error": "Training not completed"}), 400
    
    # Prepare complete results
    results = {
        "job_id": job.id,
        "filename": job.filename,
        "training_parameters": {
            "episodes": job.episodes,
            "learning_rate": job.learning_rate,
            "discount_factor": job.discount_factor,
            "epsilon": job.epsilon,
            "epsilon_decay": job.epsilon_decay,
            "initial_balance": job.initial_balance
        },
        "results": {
            "final_balance": job.final_balance,
            "total_reward": job.total_reward,
            "profit_loss": job.final_balance - job.initial_balance if job.final_balance else 0,
            "return_percentage": ((job.final_balance - job.initial_balance) / job.initial_balance * 100) if job.final_balance else 0
        },
        "portfolio_history": json.loads(job.portfolio_history) if job.portfolio_history else [],
        "reward_history": json.loads(job.reward_history) if job.reward_history else [],
        "timestamps": {
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "created_at": job.created_at.isoformat()
        }
    }
    
    # Create JSON response with download headers
    response = Response(
        json.dumps(results, indent=2),
        mimetype='application/json',
        headers={
            'Content-Disposition': f'attachment; filename=training_results_{job_id}.json'
        }
    )
    
    return response


# ==================== RL AGENT ROUTES ====================

@bp.route('/rl-agent/<int:user_id>/state', methods=['GET'])
def get_rl_state(user_id):
    """Get Q-Learning agent state"""
    state_key = request.args.get('state_key')
    
    if state_key:
        state = RLAgentState.query.filter_by(user_id=user_id, state_key=state_key).first()
        if state:
            return jsonify(state.to_dict())
        return jsonify({'message': 'State not found'}), 404
    else:
        # Get all states
        states = RLAgentState.query.filter_by(user_id=user_id).all()
        return jsonify([s.to_dict() for s in states])


@bp.route('/rl-agent/<int:user_id>/state', methods=['POST'])
def update_rl_state(user_id):
    """Update Q-Learning agent state"""
    data = request.json
    
    state = RLAgentState.query.filter_by(
        user_id=user_id, 
        state_key=data['state_key']
    ).first()
    
    if state:
        state.q_buy = data.get('q_buy', state.q_buy)
        state.q_hold = data.get('q_hold', state.q_hold)
        state.q_sell = data.get('q_sell', state.q_sell)
        state.visit_count += 1
    else:
        state = RLAgentState(
            user_id=user_id,
            state_key=data['state_key'],
            q_buy=data.get('q_buy', 0),
            q_hold=data.get('q_hold', 0),
            q_sell=data.get('q_sell', 0),
            visit_count=1
        )
        db.session.add(state)
    
    db.session.commit()
    return jsonify({'message': 'State updated', 'state': state.to_dict()})


@bp.route('/rl-agent/load/<int:job_id>', methods=['GET'])
def load_q_table(job_id):
    """Load trained Q-table from a completed job"""
    job = RLTrainingJob.query.get_or_404(job_id)
    
    if job.status != 'completed':
        return jsonify({'error': 'Training not completed'}), 400
    
    try:
        # Try loading from pickle first (faster)
        if job.q_table_path and os.path.exists(job.q_table_path):
            with open(job.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
                # Unwrap if it's the dictionary we saved with metadata
                if isinstance(q_table, dict) and 'q_table' in q_table:
                    q_table = q_table['q_table']
            
            return jsonify({
                'message': 'Q-table loaded from pickle',
                'job_id': job_id,
                'states_count': len(q_table),
                'q_table': {str(k): v for k, v in list(q_table.items())[:100]}  # Return first 100 for preview
            })
        
        # Fallback: load from database
        states = RLAgentState.query.filter_by(job_id=job_id).all()
        
        if not states:
            return jsonify({'error': 'No Q-table data found'}), 404
        
        q_table = {}
        for state in states:
            q_table[state.state_key] = [state.q_buy, state.q_hold, state.q_sell]
        
        return jsonify({
            'message': 'Q-table loaded from database',
            'job_id': job_id,
            'states_count': len(q_table),
            'q_table_preview': {k: v for k, v in list(q_table.items())[:100]}
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load Q-table: {str(e)}'}), 500


@bp.route('/rl-agent/<int:job_id>/predict', methods=['POST'])
def predict_action(job_id):
    """Use trained Q-table to predict best action for given state"""
    job = RLTrainingJob.query.get_or_404(job_id)
    
    if job.status != 'completed':
        return jsonify({'error': 'Training not completed'}), 400
    
    data = request.json
    state_key = data.get('state_key')
    
    if not state_key:
        return jsonify({'error': 'state_key required'}), 400
    
    try:
        # Load Q-table from pickle
        if job.q_table_path and os.path.exists(job.q_table_path):
            with open(job.q_table_path, 'rb') as f:
                q_table = pickle.load(f)
                if isinstance(q_table, dict) and 'q_table' in q_table:
                    # In agent.save(), we save the whole dict.
                    # Q-table is at key 'q_table'.
                     q_table = q_table['q_table']
            
            # Get Q-values for this state (string key)
            q_values = q_table.get(state_key, [0.0, 0.0, 0.0])
            best_action = int(np.argmax(q_values))
            
            action_names = ['BUY', 'HOLD', 'SELL']
            
            return jsonify({
                'state_key': state_key,
                'q_values': {
                    'buy': float(q_values[1]),  # 1
                    'hold': float(q_values[0]), # 0
                    'sell': float(q_values[2])  # 2
                },
                'best_action': action_names[best_action],
                'confidence': float(max(q_values))
            })
        else:
            return jsonify({'error': 'Q-table pickle not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
