from flask import Blueprint, render_template, jsonify
from backend.models.user import User
from backend.models.trading import Trade, Holding, Watchlist

bp = Blueprint('main', __name__)

# Handles both / and /index.html
@bp.route('/')
@bp.route('/index.html') 
def index_page():
    """Serves the main dashboard page."""
    return render_template('index.html')

# Handles both /trading and /trading.html
@bp.route('/trading')
@bp.route('/trading.html') 
def trading_page():
    """Serves the advanced trading page."""
    return render_template('trading.html')

# Handles both /q_learning and /q_learning.html
@bp.route('/Q_learning')
@bp.route('/Q_learning.html') 
def q_learning_page():
    """Serves the Q-Learning analysis page."""
    return render_template('Q_learning.html')

# Handles both /news and /news.html
@bp.route('/news')
@bp.route('/news.html') 
def news_page():
    """Serves the market news page."""
    return render_template('news.html')

# Handles both /portfolio and /portfolio.html
@bp.route('/portfolio')
@bp.route('/portfolio.html') 
def portfolio_page():
    """Serves the portfolio overview page."""
    return render_template('portfolio.html')

# Handles both /analytics and /analytics.html
@bp.route('/analytics')
@bp.route('/analytics.html') 
def analytics_page():
    """Serves the portfolio analytics page."""
    return render_template('analytics.html')

# Handles both /settings and /settings.html
@bp.route('/settings')
@bp.route('/settings.html') 
def settings_page():
    """Serves the settings page."""
    return render_template('settings.html')

@bp.route('/rl_training')
@bp.route('/rl_training.html') 
def rl_training_page():
    """Serves the RL Agent Training page."""
    return render_template('rl_training.html')

# -------- UTILITY ROUTES --------

@bp.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    from datetime import datetime
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'timestamp': datetime.utcnow().isoformat()
    })

@bp.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify({
        'total_users': User.query.count(),
        'total_trades': Trade.query.count(),
        'total_holdings': Holding.query.count(),
        'total_watchlist_items': Watchlist.query.count()
    })
