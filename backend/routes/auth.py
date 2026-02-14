from flask import Blueprint, jsonify, request
from backend.models.user import User
from backend.extensions import db

bp = Blueprint('auth', __name__)

@bp.route('/users', methods=['GET'])
def get_users():
    """Get all users"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])

@bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())

@bp.route('/users/<int:user_id>/balance', methods=['PUT'])
def update_balance(user_id):
    """Update user cash balance"""
    user = User.query.get_or_404(user_id)
    data = request.json
    user.cash_balance = data.get('cash_balance', user.cash_balance)
    db.session.commit()
    return jsonify({'message': 'Balance updated', 'user': user.to_dict()})
