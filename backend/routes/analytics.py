from flask import Blueprint, jsonify, request
from backend.models.analytics import PortfolioMetrics
from backend.extensions import db

bp = Blueprint('analytics', __name__)

@bp.route('/metrics/<int:user_id>', methods=['GET'])
def get_portfolio_metrics(user_id):
    """Get latest portfolio metrics"""
    metrics = PortfolioMetrics.query.filter_by(user_id=user_id).order_by(
        PortfolioMetrics.timestamp.desc()
    ).first()
    
    if metrics:
        return jsonify(metrics.to_dict())
    return jsonify({'message': 'No metrics found'}), 404


@bp.route('/metrics/<int:user_id>/history', methods=['GET'])
def get_metrics_history(user_id):
    """Get historical portfolio metrics"""
    days = request.args.get('days', 30, type=int)
    metrics = PortfolioMetrics.query.filter_by(user_id=user_id).order_by(
        PortfolioMetrics.timestamp.desc()
    ).limit(days).all()
    
    return jsonify([m.to_dict() for m in metrics])


@bp.route('/metrics/<int:user_id>', methods=['POST']) 
def save_portfolio_metrics(user_id):
    """Save portfolio metrics snapshot"""
    data = request.json
    
    metrics = PortfolioMetrics(
        user_id=user_id,
        total_value=data['total_value'],
        cash_balance=data['cash_balance'],
        invested_amount=data['invested_amount'],
        day_change=data.get('day_change', 0),
        day_change_percent=data.get('day_change_percent', 0),
        sharpe_ratio=data.get('sharpe_ratio'),
        beta=data.get('beta'),
        alpha=data.get('alpha'),
        volatility=data.get('volatility'),
        max_drawdown=data.get('max_drawdown'),
        win_rate=data.get('win_rate')
    )
    
    db.session.add(metrics)
    db.session.commit()
    
    return jsonify({'message': 'Metrics saved', 'metrics': metrics.to_dict()}), 201
