from flask import Blueprint, jsonify, request
from backend.models.user import User
from backend.models.trading import Holding, Trade, Watchlist
from backend.extensions import db
from sqlalchemy.exc import IntegrityError

bp = Blueprint('trading', __name__)

# -------- HOLDINGS ROUTES --------

@bp.route('/holdings/<int:user_id>', methods=['GET'])
def get_holdings(user_id):
    """Get all holdings for a user"""
    holdings = Holding.query.filter_by(user_id=user_id).all()
    return jsonify([holding.to_dict() for holding in holdings])


@bp.route('/holdings/<int:user_id>/<symbol>', methods=['GET'])
def get_holding(user_id, symbol):
    """Get specific holding"""
    holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    return jsonify(holding.to_dict())


@bp.route('/holdings/<int:user_id>', methods=['POST'])
def add_holding(user_id):
    """Add or update a holding"""
    data = request.json
    
    existing = Holding.query.filter_by(
        user_id=user_id, 
        symbol=data['symbol']
    ).first()
    
    if existing:
        # Update existing holding
        existing.shares += data.get('shares', 0)
        # Recalculate average cost
        total_cost = (existing.shares * existing.avg_cost) + (data['shares'] * data['price'])
        existing.avg_cost = total_cost / existing.shares
        existing.current_price = data.get('current_price', data['price'])
        db.session.commit()
        return jsonify({'message': 'Holding updated', 'holding': existing.to_dict()})
    else:
        # Create new holding
        holding = Holding(
            user_id=user_id,
            symbol=data['symbol'],
            company_name=data.get('company_name'),
            shares=data['shares'],
            avg_cost=data['price'],
            current_price=data.get('current_price', data['price'])
        )
        db.session.add(holding)
        db.session.commit()
        return jsonify({'message': 'Holding created', 'holding': holding.to_dict()}), 201


@bp.route('/holdings/<int:user_id>/<symbol>', methods=['PUT'])
def update_holding(user_id, symbol):
    """Update holding price (can still be used for manual or external API pushes)"""
    holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    data = request.json
    holding.current_price = data.get('current_price', holding.current_price)
    db.session.commit()
    return jsonify({'message': 'Holding updated', 'holding': holding.to_dict()})


@bp.route('/holdings/<int:user_id>/<symbol>', methods=['DELETE'])
def delete_holding(user_id, symbol):
    """Delete a holding"""
    holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    db.session.delete(holding)
    db.session.commit()
    return jsonify({'message': 'Holding deleted'})


# -------- TRADES ROUTES --------

@bp.route('/trades/<int:user_id>', methods=['GET'])
def get_trades(user_id):
    """Get all trades for a user"""
    trades = Trade.query.filter_by(user_id=user_id).order_by(Trade.timestamp.desc()).all()
    return jsonify([trade.to_dict() for trade in trades])


@bp.route('/trades/<int:user_id>', methods=['POST'])
def execute_trade(user_id):
    """Execute a new trade"""
    data = request.json
    from backend.services.trading_engine import TradingEngine
    
    result = TradingEngine.execute_trade(user_id, data, dry_run=False)
    
    if result.get('status') == 'error':
        return jsonify({'error': result.get('error')}), 400
        
    return jsonify({
        'message': result.get('message'),
        'trade': result.get('trade'),
        'new_balance': result.get('new_balance')
    }), 201


# -------- WATCHLIST ROUTES --------

@bp.route('/watchlist/<int:user_id>', methods=['GET'])
def get_watchlist(user_id):
    """Get user's watchlist"""
    watchlist = Watchlist.query.filter_by(user_id=user_id).all()
    return jsonify([item.to_dict() for item in watchlist])


@bp.route('/watchlist/<int:user_id>', methods=['POST'])
def add_to_watchlist(user_id):
    """Add symbol to watchlist"""
    data = request.json
    
    try:
        watchlist_item = Watchlist(
            user_id=user_id,
            symbol=data['symbol'],
            company_name=data.get('company_name')
        )
        db.session.add(watchlist_item)
        db.session.commit()
        return jsonify({'message': 'Added to watchlist', 'item': watchlist_item.to_dict()}), 201
    except IntegrityError:
        db.session.rollback()
        return jsonify({'error': 'Symbol already in watchlist'}), 400


@bp.route('/watchlist/<int:user_id>/<symbol>', methods=['DELETE'])
def remove_from_watchlist(user_id, symbol):
    """Remove symbol from watchlist"""
    item = Watchlist.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    db.session.delete(item)
    db.session.commit()
    return jsonify({'message': 'Removed from watchlist'})
