

from flask import Flask, jsonify, request, render_template ,send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, UTC
from sqlalchemy.exc import IntegrityError
import os
import pandas as pd
import numpy as np
import json
from werkzeug.utils import secure_filename
import subprocess
import threading
# -*- coding: utf-8 -*-
import sys
import requests
from apscheduler.schedulers.background import BackgroundScheduler # New Import
import time
# Note: BackgroundScheduler is used here for simplicity. For production, Flask-APScheduler or Celery is recommended.
import matplotlib
import pickle 
matplotlib.use('Agg') # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
# Force UTF-8 for Windows consoles that default to cp1252
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Initialize Flask app
app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# ✅FILE UPLOAD CONFIG
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# ✅ END FILE UPLOAD CONFIG
MODELS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_FOLDER, exist_ok=True)
# New: External API Configuration (Get key from environment or use placeholder)
FINNHUB_API_KEY = 'd3f9q79r01qolknc9vhgd3f9q79r01qolknc9vi0'
NEWS_API_KEY = '1acb4f9a6da041d4acde2a75072f16d7'
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1' # ***REPLACE WITH YOUR ACTUAL KEY***
FINNHUB_BASE_URL = 'https://finnhub.io/api/v1'

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 
    'postgresql://ai_trading_pro:****@localhost:5432/ai_trading_db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'

# Initialize SQLAlchemy
db = SQLAlchemy(app)


class User(db.Model):
    """User accounts table"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    cash_balance = db.Column(db.Float, default=15670.25)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    holdings = db.relationship('Holding', backref='user', lazy=True, cascade='all, delete-orphan')
    trades = db.relationship('Trade', backref='user', lazy=True, cascade='all, delete-orphan')
    watchlists = db.relationship('Watchlist', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'cash_balance': self.cash_balance,
            'created_at': self.created_at.isoformat(),
        }


class Holding(db.Model):
    """Portfolio holdings table"""
    __tablename__ = 'holdings'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(200))
    shares = db.Column(db.Integer, nullable=False)
    avg_cost = db.Column(db.Float, nullable=False)
    current_price = db.Column(db.Float, default=0.0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint: one holding per symbol per user
    __table_args__ = (db.UniqueConstraint('user_id', 'symbol', name='_user_symbol_uc'),)
    
    def to_dict(self):
        market_value = self.shares * self.current_price
        total_cost = self.shares * self.avg_cost
        gain_loss = market_value - total_cost
        gain_loss_percent = (gain_loss / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'id': self.id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'shares': self.shares,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': market_value,
            'gain_loss': gain_loss,
            'gain_loss_percent': gain_loss_percent,
            'last_updated': self.last_updated.isoformat(),
        
        }


class Trade(db.Model):
    """Trade history table"""
    __tablename__ = 'trades'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # BUY, SELL
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    order_type = db.Column(db.String(20), default='market')  # market, limit, stop
    time_in_force = db.Column(db.String(10), default='day')  # day, gtc, ioc, fok
    status = db.Column(db.String(20), default='completed')  # completed, pending, cancelled
    total_value = db.Column(db.Float, nullable=False)
    commission = db.Column(db.Float, default=0.0)
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'order_type': self.order_type,
            'time_in_force': self.time_in_force,
            'status': self.status,
            'total_value': self.total_value,
            'commission': self.commission,
            'notes': self.notes,
            'timestamp': self.timestamp.isoformat()
        }


class Watchlist(db.Model):
    """User watchlist table"""
    __tablename__ = 'watchlists'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(200))
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint: one watchlist entry per symbol per user
    __table_args__ = (db.UniqueConstraint('user_id', 'symbol', name='_user_watchlist_uc'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'added_at': self.added_at.isoformat()
        }


class PortfolioMetrics(db.Model):
    """Historical portfolio metrics for analytics"""
    __tablename__ = 'portfolio_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    total_value = db.Column(db.Float, nullable=False)
    cash_balance = db.Column(db.Float, nullable=False)
    invested_amount = db.Column(db.Float, nullable=False)
    day_change = db.Column(db.Float, default=0.0)
    day_change_percent = db.Column(db.Float, default=0.0)
    sharpe_ratio = db.Column(db.Float)
    beta = db.Column(db.Float)
    alpha = db.Column(db.Float)
    volatility = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'invested_amount': self.invested_amount,
            'day_change': self.day_change,
            'day_change_percent': self.day_change_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'beta': self.beta,
            'alpha': self.alpha,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'timestamp': self.timestamp.isoformat()
        }


class RLAgentState(db.Model):
    """Q-Learning agent state storage"""
    __tablename__ = 'rl_agent_states'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('rl_training_jobs.id'))
    state_key = db.Column(db.String(100), nullable=False)
    q_buy = db.Column(db.Float, default=0.0)
    q_hold = db.Column(db.Float, default=0.0)
    q_sell = db.Column(db.Float, default=0.0)
    visit_count = db.Column(db.Integer, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (db.UniqueConstraint('user_id', 'job_id', 'state_key', name='_user_state_uc'),)
    
    def to_dict(self):
        return {
            'id': self.id,
            'job_id': self.job_id,
            'state_key': self.state_key,
            'q_buy': self.q_buy,
            'q_hold': self.q_hold,
            'q_sell': self.q_sell,
            'visit_count': self.visit_count,
            'last_updated': self.last_updated.isoformat(),
        }

class RLTrainingJob(db.Model):
    """Track RL agent training jobs"""
    __tablename__ = 'rl_training_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed
    episodes = db.Column(db.Integer, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    discount_factor = db.Column(db.Float, nullable=False)
    epsilon = db.Column(db.Float, nullable=False)
    epsilon_decay = db.Column(db.Float, nullable=False)
    initial_balance = db.Column(db.Float, nullable=False)
    progress = db.Column(db.Integer, default=0)  # 0-100%
    final_balance = db.Column(db.Float)
    total_reward = db.Column(db.Float)
    error_message = db.Column(db.Text)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolio_history = db.Column(db.Text)
    reward_history = db.Column(db.Text)
    q_table_path = db.Column(db.String(500))
    def to_dict(self):
        result = {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'status': self.status,
            'episodes': self.episodes,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'initial_balance': self.initial_balance,
            'progress': self.progress,
            'final_balance': self.final_balance,
            'total_reward': self.total_reward,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat(),
            'q_table_path': self.q_table_path,
        }
        
        # Include history data if available
        if self.portfolio_history:
            try:
                result['portfolio_history'] = json.loads(self.portfolio_history)
            except:
                result['portfolio_history'] = []
        
        if self.reward_history:
            try:
                result['reward_history'] = json.loads(self.reward_history)
            except:
                result['reward_history'] = []
        
        return result

# ==================== LIVE DATA SCHEDULER FUNCTIONS (No Changes) ====================

def fetch_current_price(symbol):
    """Fetches the latest real-time stock price from Finnhub."""
    # Check for API key presence
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == 'YOUR_FINNHUB_API_KEY_HERE':
        print(f"⚠️ WARNING: Finnhub API key is not set. Cannot fetch live data for {symbol}.")
        return None

    # Finnhub 'quote' endpoint
    url = f"{FINNHUB_BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        # 'c' is the current price
        if data and data.get('c') is not None and data.get('c') > 0:
            current_price = data['c']
            return current_price
        else:
            print(f"🛑 Error: Invalid or missing data for {symbol} from Finnhub API. Response: {data}")
            return None
    except requests.exceptions.RequestException as e:
        # Catch network errors or HTTP errors
        print(f"❌ Error fetching live data for {symbol}: {e}")
        return None

def update_all_holdings_prices():
    """Scheduled task to update the current price for all unique holdings."""
    # This must run within the Flask application context to interact with the database
    with app.app_context():
        print(f"\n🔄 Running scheduled price update at {datetime.now().strftime('%H:%M:%S')}...")
        
        # Get all unique symbols currently held by any user
        unique_symbols = db.session.query(Holding.symbol).distinct().all()
        symbols_to_update = [s[0] for s in unique_symbols]
        
        if not symbols_to_update:
            print("ℹ️ No holdings found to update.")
            return

        print(f"Updating prices for symbols: {symbols_to_update}")
        
        # Fetch prices for all symbols
        for symbol in symbols_to_update:
            live_price = fetch_current_price(symbol)
            
            if live_price is not None:
                # Update all holdings with this symbol across all users
                updated_count = db.session.query(Holding).filter(
                    Holding.symbol == symbol
                ).update(
                    {'current_price': live_price, 'last_updated': datetime.utcnow()},
                    synchronize_session='fetch'
                )
                if updated_count > 0:
                    print(f"   -> Successfully updated {updated_count} holding(s) for {symbol} to ${live_price:.2f}")
        
        try:
            db.session.commit()
            print("✅ Database commit successful.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Database transaction failed during price update: {e}")


# ==================== DATABASE INITIALIZATION (No Changes) ====================

def init_db():
    """Initialize database and create all tables"""
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")
        
        # Create demo user if not exists
        demo_user = User.query.filter_by(username='demo_user').first()
        if not demo_user:
            demo_user = User(
                username='demo_user',
                email='demo@aitrading.pro',
                password_hash='demo_password_hash',
                cash_balance=15670.25
            )
            db.session.add(demo_user)
            
            # Add demo holdings (Removed static current_price, as the scheduler will set it)
            demo_holdings = [
                Holding(user_id=1, symbol='AAPL', company_name='Apple Inc.', 
                       shares=100, avg_cost=150.25),
                Holding(user_id=1, symbol='TSLA', company_name='Tesla Inc.', 
                       shares=50, avg_cost=220.80),
                Holding(user_id=1, symbol='GOOGL', company_name='Alphabet Inc.', 
                       shares=25, avg_cost=2800.40),
                Holding(user_id=1, symbol='MSFT', company_name='Microsoft Corp.', 
                       shares=75, avg_cost=280.60)
            ]
            for holding in demo_holdings:
                db.session.add(holding)
            
            # Add demo watchlist
            demo_watchlist = [
                Watchlist(user_id=1, symbol='AAPL', company_name='Apple Inc.'),
                Watchlist(user_id=1, symbol='TSLA', company_name='Tesla Inc.'),
                Watchlist(user_id=1, symbol='NVDA', company_name='NVIDIA Corp.'),
                Watchlist(user_id=1, symbol='AMZN', company_name='Amazon Inc.')
            ]
            for item in demo_watchlist:
                db.session.add(item)
            
            db.session.commit()
            print("✅ Demo user and data created successfully!")


# ==================== ADD UTILITY FUNCTIONS (before routes) ====================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_csv_data(filepath):
    """Validate CSV file structure"""
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        if 'Close' not in df.columns:
            return False, "CSV must contain a 'Close' column"
        
        # Validate Close column
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if df['Close'].isna().all():
            return False, "Close column must contain numeric values"
        
        # Check for sufficient data
        if len(df) < 10:
            return False, "CSV must contain at least 10 rows of data"
        
        return True, "Valid CSV file"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"

def run_rl_training(job_id, filepath, params):
    """Run REAL RL training using actual CSV data and Q-Learning"""
    
    # Import the REAL implementations from utils folder
    import sys
    import os
    
    # Add utils directory to Python path
    utils_path = os.path.join(os.path.dirname(__file__), 'utils')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    
    from environment import TradingEnv
    from agent import QLearningAgent

    with app.app_context():
        job = db.session.get(RLTrainingJob, job_id)
        if not job:
            print(f"[RL ERROR] Job {job_id} not found")
            return
        
        try:
            print(f"[RL] 🚀 Starting REAL training for job {job_id}")
            job.status = "running"
            job.started_at = datetime.now(UTC)
            db.session.commit()

            # ========== LOAD REAL CSV DATA ==========
            print(f"[RL] 📊 Loading CSV data from {filepath}")
            df = pd.read_csv(filepath)
            
            # Validate CSV has Close column
            if 'Close' not in df.columns:
                raise ValueError("CSV must contain 'Close' column")
            
            # Extract actual price data
            prices = df["Close"].values.tolist()
            print(f"[RL] ✅ Loaded {len(prices)} price points from CSV")
            print(f"[RL]    Price range: ${min(prices):.2f} - ${max(prices):.2f}")

            # ========== INITIALIZE REAL ENVIRONMENT ==========
            env = TradingEnv(
                data=prices,
                initial_balance=params["initial_balance"],
                tx_cost=0.001  # 0.1% transaction cost
            )
            print(f"[RL] 🏗️  Environment initialized with ${params['initial_balance']} balance")

            # ========== INITIALIZE REAL Q-LEARNING AGENT ==========
            agent = QLearningAgent(
                action_size=3,
                learning_rate=params["learning_rate"],
                discount_factor=params["discount_factor"],
                exploration_rate=params["epsilon"],
                exploration_decay=params["epsilon_decay"],
                min_exploration=0.01
            )
            print(f"[RL] 🤖 Agent initialized:")
            print(f"[RL]    Learning rate (α): {params['learning_rate']}")
            print(f"[RL]    Discount factor (γ): {params['discount_factor']}")
            print(f"[RL]    Exploration rate (ε): {params['epsilon']}")

            episodes = params["episodes"]
            portfolio_history = []
            reward_history = []

            # ========== REAL TRAINING LOOP ==========
            print(f"[RL] 🏋️  Starting training for {episodes} episodes...")
            
            for episode in range(1, episodes + 1):
                state = env.reset()
                done = False
                total_reward = 0
                steps = 0

                # Run episode until done
                while not done:
                    # Agent chooses action using epsilon-greedy
                    action = agent.choose_action(state)
                    
                    # Environment executes action
                    next_state, reward, done, info = env.step(action)
                    
                    # Agent learns from experience
                    agent.learn(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                # Get final portfolio value
                final_price = prices[-1]
                final_portfolio_value = info['portfolio_value']
                
                portfolio_history.append(final_portfolio_value)
                reward_history.append(total_reward)

                # Decay exploration rate
                agent.decay_epsilon()

                # Progress logging
                if episode % 50 == 0 or episode == episodes:
                    log_msg = f"[RL] Episode {episode}/{episodes}: " \
                             f"Value=${final_portfolio_value:.2f}, " \
                             f"Reward={total_reward:.2f}, " \
                             f"Trades={info['total_trades']}, " \
                             f"ε={agent.epsilon:.3f}, " \
                             f"States={len(agent.q_table)}"
                    print(log_msg)

                # Update job progress
                job.progress = int((episode / episodes) * 100)
                if episode % 10 == 0:  # Commit every 10 episodes
                    db.session.commit()

            print(f"[RL] ✅ Training completed!")
            print(f"[RL]    Total Q-table states: {len(agent.q_table)}")

            # ========== SAVE Q-TABLE TO DATABASE ==========
            print(f"[RL] 💾 Saving Q-table to database...")
            
            # Clear old states for this job
            try:
                deleted = db.session.query(RLAgentState).filter_by(
                    user_id=job.user_id,
                    job_id=job_id
                ).delete()
                db.session.commit()
                print(f"[RL]    Cleared {deleted} old states")
            except Exception as e:
                print(f"[RL WARNING] Could not clear old states: {e}")
                db.session.rollback()
            
            # Save new states in batches
            saved_count = 0
            batch_size = 100
            states_to_add = []
            
            for state_key, q_values in agent.q_table.items():
                try:
                    rl_state = RLAgentState(
                        user_id=job.user_id,
                        job_id=job_id,
                        state_key=str(state_key),
                        q_buy=float(q_values[1]),      # Action 1 = BUY
                        q_hold=float(q_values[0]),     # Action 0 = HOLD
                        q_sell=float(q_values[2]),     # Action 2 = SELL
                        visit_count=1
                    )
                    states_to_add.append(rl_state)
                    saved_count += 1
                    
                    if len(states_to_add) >= batch_size:
                        db.session.bulk_save_objects(states_to_add)
                        db.session.commit()
                        states_to_add = []
                        
                except Exception as e:
                    print(f"[RL ERROR] Failed to save state {state_key}: {e}")
                    db.session.rollback()
                    continue
            
            # Save remaining states
            if states_to_add:
                db.session.bulk_save_objects(states_to_add)
                db.session.commit()
            
            print(f"[RL] ✅ Saved {saved_count} Q-table states to database")

            # ========== SAVE Q-TABLE AS PICKLE ==========
            try:
                models_dir = os.path.join(os.path.dirname(__file__), 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                q_table_filename = f'q_table_job_{job_id}.pkl'
                q_table_path = os.path.join(models_dir, q_table_filename)
                
                agent.save(q_table_path)
                job.q_table_path = q_table_path
                print(f"[RL] ✅ Q-table pickle saved to {q_table_path}")
                
            except Exception as e:
                print(f"[RL ERROR] Failed to save pickle: {e}")
                job.error_message = f"Pickle save failed: {str(e)}"

            # ========== SAVE FINAL RESULTS ==========
            final_value = portfolio_history[-1]
            total_reward = reward_history[-1]
            
            job.final_balance = float(final_value)
            job.total_reward = float(sum(reward_history))
            job.portfolio_history = json.dumps(portfolio_history)
            job.reward_history = json.dumps(reward_history)

            # Save results JSON file
            try:
                results_dir = os.path.join(os.path.dirname(__file__), "results")
                os.makedirs(results_dir, exist_ok=True)
                
                results_data = {
                    "job_id": job_id,
                    "episodes": episodes,
                    "initial_balance": params["initial_balance"],
                    "final_balance": job.final_balance,
                    "total_reward": job.total_reward,
                    "profit_loss": job.final_balance - params["initial_balance"],
                    "return_pct": ((job.final_balance - params["initial_balance"]) / params["initial_balance"] * 100),
                    "portfolio_history": portfolio_history,
                    "reward_history": reward_history,
                    "q_table_states": saved_count,
                    "training_params": params
                }
                
                results_path = os.path.join(results_dir, f"results_{job_id}.json")
                with open(results_path, "w") as f:
                    json.dump(results_data, f, indent=4)
                    
                print(f"[RL] ✅ Results saved to {results_path}")

            except Exception as e:
                print(f"[RL ERROR] Failed to save results: {e}")
            
            # ========== GENERATE PLOT ==========
            try:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.plot(portfolio_history, color='#00f5ff', linewidth=2)
                plt.axhline(y=params["initial_balance"], color='red', 
                           linestyle='--', label='Initial Balance')
                plt.title(f'Portfolio Value Over Episodes (Job {job_id})')
                plt.xlabel('Episode')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(reward_history, color='#ff00ff', linewidth=2)
                plt.title('Cumulative Reward Per Episode')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(results_dir, 'trading_plot.png')
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                print(f"[RL] ✅ Plot saved to {plot_path}")

            except Exception as e:
                print(f"[RL ERROR] Failed to generate plot: {e}")

            # ========== UPDATE JOB STATUS ==========
            job.status = "completed"
            job.progress = 100
            job.completed_at = datetime.now(UTC)
            db.session.commit()
            
            print(f"[RL] 🎉 Training complete for job {job_id}!")
            print(f"[RL]    Initial: ${params['initial_balance']:.2f}")
            print(f"[RL]    Final: ${job.final_balance:.2f}")
            print(f"[RL]    Profit/Loss: ${job.final_balance - params['initial_balance']:.2f}")
            print(f"[RL]    Return: {((job.final_balance - params['initial_balance']) / params['initial_balance'] * 100):.2f}%")
            
        except Exception as e:
            print(f"[RL ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            
            job.status = "failed"
            job.error_message = str(e)
            db.session.rollback()
            db.session.commit()

def migrate_rl_training_jobs():
    """Add missing columns to tables"""
    with app.app_context():
        try:
            from sqlalchemy import text, inspect
            
            # Check if columns exist first
            inspector = inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('rl_training_jobs')]
            
            # Add portfolio_history column
            if 'portfolio_history' not in existing_columns:
                try:
                    db.session.execute(text(
                        "ALTER TABLE rl_training_jobs ADD COLUMN portfolio_history TEXT"
                    ))
                    db.session.commit()
                    print("✅ Added portfolio_history column")
                except Exception as e:
                    db.session.rollback()
                    print(f"⚠️ portfolio_history column error: {e}")
            else:
                print("ℹ️ portfolio_history column already exists")
            
            # Add reward_history column
            if 'reward_history' not in existing_columns:
                try:
                    db.session.execute(text(
                        "ALTER TABLE rl_training_jobs ADD COLUMN reward_history TEXT"
                    ))
                    db.session.commit()
                    print("✅ Added reward_history column")
                except Exception as e:
                    db.session.rollback()
                    print(f"⚠️ reward_history column error: {e}")
            else:
                print("ℹ️ reward_history column already exists")
            
            # Add q_table_path column - THIS IS THE CRITICAL ONE
            if 'q_table_path' not in existing_columns:
                try:
                    db.session.execute(text(
                        "ALTER TABLE rl_training_jobs ADD COLUMN q_table_path VARCHAR(500)"
                    ))
                    db.session.commit()
                    print("✅ Added q_table_path column")
                except Exception as e:
                    db.session.rollback()
                    print(f"❌ q_table_path column error: {e}")
            else:
                print("ℹ️ q_table_path column already exists")
            
            # Check rl_agent_states table
            agent_columns = [col['name'] for col in inspector.get_columns('rl_agent_states')]
            
            # Add job_id column to rl_agent_states
            if 'job_id' not in agent_columns:
                try:
                    db.session.execute(text(
                        "ALTER TABLE rl_agent_states ADD COLUMN job_id INTEGER REFERENCES rl_training_jobs(id)"
                    ))
                    db.session.commit()
                    print("✅ Added job_id column to rl_agent_states")
                except Exception as e:
                    db.session.rollback()
                    print(f"⚠️ job_id column error: {e}")
            else:
                print("ℹ️ job_id column already exists in rl_agent_states")
            
            print("✅ Database migration completed")
            
        except Exception as e:
            print(f"❌ Migration error: {e}")
            db.session.rollback()

# -------- USER ROUTES --------

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users"""
    users = User.query.all()
    return jsonify([user.to_dict() for user in users])


@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    """Get specific user"""
    user = User.query.get_or_404(user_id)
    return jsonify(user.to_dict())


@app.route('/api/users/<int:user_id>/balance', methods=['PUT'])
def update_balance(user_id):
    """Update user cash balance"""
    user = User.query.get_or_404(user_id)
    data = request.json
    user.cash_balance = data.get('cash_balance', user.cash_balance)
    db.session.commit()
    return jsonify({'message': 'Balance updated', 'user': user.to_dict()})


# -------- HOLDINGS ROUTES --------

@app.route('/api/holdings/<int:user_id>', methods=['GET'])
def get_holdings(user_id):
    """Get all holdings for a user"""
    holdings = Holding.query.filter_by(user_id=user_id).all()
    return jsonify([holding.to_dict() for holding in holdings])


@app.route('/api/holdings/<int:user_id>/<symbol>', methods=['GET'])
def get_holding(user_id, symbol):
    """Get specific holding"""
    holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    return jsonify(holding.to_dict())


@app.route('/api/holdings/<int:user_id>', methods=['POST'])
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


@app.route('/api/holdings/<int:user_id>/<symbol>', methods=['PUT'])
def update_holding(user_id, symbol):
    """Update holding price (can still be used for manual or external API pushes)"""
    holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    data = request.json
    holding.current_price = data.get('current_price', holding.current_price)
    db.session.commit()
    return jsonify({'message': 'Holding updated', 'holding': holding.to_dict()})


@app.route('/api/holdings/<int:user_id>/<symbol>', methods=['DELETE'])
def delete_holding(user_id, symbol):
    """Delete a holding"""
    holding = Holding.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    db.session.delete(holding)
    db.session.commit()
    return jsonify({'message': 'Holding deleted'})


# -------- TRADES ROUTES (No Changes) --------

@app.route('/api/trades/<int:user_id>', methods=['GET'])
def get_trades(user_id):
    """Get all trades for a user"""
    trades = Trade.query.filter_by(user_id=user_id).order_by(Trade.timestamp.desc()).all()
    return jsonify([trade.to_dict() for trade in trades])


@app.route('/api/trades/<int:user_id>', methods=['POST'])
def execute_trade(user_id):
    """Execute a new trade"""
    data = request.json
    user = User.query.get_or_404(user_id)
    
    # Validate trade data
    if not all(k in data for k in ['symbol', 'action', 'quantity', 'price']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    action = data['action'].upper()
    quantity = int(data['quantity'])
    price = float(data['price'])
    total_value = quantity * price
    commission = total_value * 0.001  # 0.1% commission
    
    # Create trade record
    trade = Trade(
        user_id=user_id,
        symbol=data['symbol'],
        action=action,
        quantity=quantity,
        price=price,
        order_type=data.get('order_type', 'market'),
        time_in_force=data.get('time_in_force', 'day'),
        total_value=total_value,
        commission=commission,
        notes=data.get('notes')
    )
    
    # Update holdings and cash
    if action == 'BUY':
        total_cost = total_value + commission
        if user.cash_balance < total_cost:
            return jsonify({'error': 'Insufficient funds'}), 400
        
        user.cash_balance -= total_cost
        
        # Update or create holding
        holding = Holding.query.filter_by(user_id=user_id, symbol=data['symbol']).first()
        if holding:
            total_shares = holding.shares + quantity
            total_cost = (holding.shares * holding.avg_cost) + (quantity * price)
            holding.avg_cost = total_cost / total_shares
            holding.shares = total_shares
            holding.current_price = price
        else:
            holding = Holding(
                user_id=user_id,
                symbol=data['symbol'],
                company_name=data.get('company_name'),
                shares=quantity,
                avg_cost=price,
                current_price=price
            )
            db.session.add(holding)
    
    elif action == 'SELL':
        holding = Holding.query.filter_by(user_id=user_id, symbol=data['symbol']).first()
        if not holding or holding.shares < quantity:
            return jsonify({'error': 'Insufficient shares'}), 400
        
        user.cash_balance += (total_value - commission)
        holding.shares -= quantity
        
        # Remove holding if all shares sold
        if holding.shares == 0:
            db.session.delete(holding)
    
    db.session.add(trade)
    db.session.commit()
    
    return jsonify({
        'message': 'Trade executed successfully',
        'trade': trade.to_dict(),
        'new_balance': user.cash_balance
    }), 201


# -------- WATCHLIST ROUTES (No Changes) --------

@app.route('/api/watchlist/<int:user_id>', methods=['GET'])
def get_watchlist(user_id):
    """Get user's watchlist"""
    watchlist = Watchlist.query.filter_by(user_id=user_id).all()
    return jsonify([item.to_dict() for item in watchlist])


@app.route('/api/watchlist/<int:user_id>', methods=['POST'])
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


@app.route('/api/watchlist/<int:user_id>/<symbol>', methods=['DELETE'])
def remove_from_watchlist(user_id, symbol):
    """Remove symbol from watchlist"""
    item = Watchlist.query.filter_by(user_id=user_id, symbol=symbol).first_or_404()
    db.session.delete(item)
    db.session.commit()
    return jsonify({'message': 'Removed from watchlist'})


# -------- PORTFOLIO METRICS ROUTES (No Changes) --------

@app.route('/api/metrics/<int:user_id>', methods=['GET'])
def get_portfolio_metrics(user_id):
    """Get latest portfolio metrics"""
    metrics = PortfolioMetrics.query.filter_by(user_id=user_id).order_by(
        PortfolioMetrics.timestamp.desc()
    ).first()
    
    if metrics:
        return jsonify(metrics.to_dict())
    return jsonify({'message': 'No metrics found'}), 404


@app.route('/api/metrics/<int:user_id>/history', methods=['GET'])
def get_metrics_history(user_id):
    """Get historical portfolio metrics"""
    days = request.args.get('days', 30, type=int)
    metrics = PortfolioMetrics.query.filter_by(user_id=user_id).order_by(
        PortfolioMetrics.timestamp.desc()
    ).limit(days).all()
    
    return jsonify([m.to_dict() for m in metrics])


@app.route('/api/metrics/<int:user_id>', methods=['POST']) 
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


# -------- RL AGENT ROUTES (No Changes) --------

@app.route('/api/rl-agent/<int:user_id>/state', methods=['GET'])
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


@app.route('/api/rl-agent/<int:user_id>/state', methods=['POST'])
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

# ==================== FRONTEND PAGE ROUTES (NEW) ====================
# Handles both / and /index.html
@app.route('/')
@app.route('/index.html') 
def index_page():
    """Serves the main dashboard page."""
    return render_template('index.html')

# Handles both /trading and /trading.html
@app.route('/trading')
@app.route('/trading.html') 
def trading_page():
    """Serves the advanced trading page."""
    return render_template('trading.html')

# Handles both /q_learning and /q_learning.html
@app.route('/Q_learning')
@app.route('/Q_learning.html') 
def q_learning_page():
    """Serves the Q-Learning analysis page."""
    return render_template('Q_learning.html')

# Handles both /news and /news.html
@app.route('/news')
@app.route('/news.html') 
def news_page():
    """Serves the market news page."""
    return render_template('news.html')

# Handles both /portfolio and /portfolio.html
@app.route('/portfolio')
@app.route('/portfolio.html') 
def portfolio_page():
    """Serves the portfolio overview page."""
    return render_template('portfolio.html')

# Handles both /analytics and /analytics.html
@app.route('/analytics')
@app.route('/analytics.html') 
def analytics_page():
    """Serves the portfolio analytics page."""
    return render_template('analytics.html')
# Handles both /settings and /settings.html
@app.route('/settings')
@app.route('/settings.html') 
def settings_page():
    """Serves the settings page from templates/settings.html."""
    return render_template('settings.html')

@app.route('/rl_training')
@app.route('/rl_training.html') 
def rl_training_page():
    """Serves the RL Agent Training page."""
    return render_template('rl_training.html')
# -------- UTILITY ROUTES (No Changes) --------

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify({
        'total_users': User.query.count(),
        'total_trades': Trade.query.count(),
        'total_holdings': Holding.query.count(),
        'total_watchlist_items': Watchlist.query.count()
    })

@app.route('/api/rl-training/upload', methods=['POST'])
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Validate CSV
        is_valid, message = validate_csv_data(filepath)
        if not is_valid:
            os.remove(filepath)
            return jsonify({'error': message}), 400
        
        # Read data preview
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



@app.route('/api/rl-training/start', methods=['POST'])
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
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
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
        
        # Start training in background thread
        params = {
            'episodes': job.episodes,
            'learning_rate': job.learning_rate,
            'discount_factor': job.discount_factor,
            'epsilon': job.epsilon,
            'epsilon_decay': job.epsilon_decay,
            'initial_balance': job.initial_balance
        }
        
        thread = threading.Thread(
            target=run_rl_training,
            args=(job.id, filepath, params)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Training started',
            'job': job.to_dict()
        }), 201
    
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@app.route('/api/rl-training/jobs/<int:user_id>', methods=['GET'])
def get_training_jobs(user_id):
    """Get all training jobs for a user"""
    jobs = RLTrainingJob.query.filter_by(user_id=user_id).order_by(
        RLTrainingJob.created_at.desc()
    ).all()
    return jsonify([job.to_dict() for job in jobs])


@app.route('/api/rl-training/job/<int:job_id>', methods=['GET'])
def get_training_job(job_id):
    """Get specific training job status"""
    job = RLTrainingJob.query.get_or_404(job_id)
    return jsonify(job.to_dict())


@app.route("/api/rl-training/job/<int:jobid>/results", methods=["GET"])
def get_training_results(jobid):
    """Fetches the training results from the job-specific JSON file."""
    # Note: Assumes RLTrainingJob model and db.session.get exist and are correctly implemented
    job = db.session.get(RLTrainingJob, jobid)
    if not job:
        return jsonify({"error": f"Job {jobid} not found"}), 404

    if job.status != "completed":
        return jsonify({"error": "Training not completed or file not ready"}), 400
    
    # 🎯 STEP 2: LOOK FOR THE JOB-SPECIFIC RESULTS FILE
    results_path = os.path.join(
        os.path.dirname(__file__), 
        "results", 
        f"results_{jobid}.json"
    )
    
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        return jsonify(results)
    else:
        # If the file is missing, something failed after training but before cleanup
        return jsonify({"error": "Results file not found on disk. Training likely failed to save data."}), 404

@app.route('/api/rl-training/job/<int:job_id>/plot', methods=['GET'])
def get_training_plot(job_id):
    """Get training performance plot"""
    job = RLTrainingJob.query.get_or_404(job_id)
    
    if job.status != 'completed':
        return jsonify({'error': 'Training not completed'}), 400
    
    plot_path = os.path.join(os.path.dirname(__file__), 'results', 'trading_plot.png')
    
    if os.path.exists(plot_path):
        return send_from_directory(
            os.path.dirname(plot_path),
            'trading_plot.png',
            mimetype='image/png'
        )
    else:
        return jsonify({'error': 'Plot file not found'}), 404


@app.route('/api/rl-training/job/<int:job_id>', methods=['DELETE'])
def delete_training_job(job_id):
    """Delete a training job"""
    job = RLTrainingJob.query.get_or_404(job_id)
    
    # Delete uploaded file if exists
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], job.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    db.session.delete(job)
    db.session.commit()
    
    return jsonify({'message': 'Training job deleted'})
# ==================== ERROR HANDLERS (No Changes) ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/rl-training/job/<int:job_id>/download', methods=['GET'])
def download_training_results(job_id):
    """Download training results as JSON file"""
    from flask import Response
    
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

@app.route('/api/rl-agent/load/<int:job_id>', methods=['GET'])
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


@app.route('/api/rl-agent/<int:job_id>/predict', methods=['POST'])
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
            
            # Get Q-values for this state
            q_values = q_table.get(state_key, [0.0, 0.0, 0.0])
            best_action = int(np.argmax(q_values))
            
            action_names = ['BUY', 'HOLD', 'SELL']
            
            return jsonify({
                'state_key': state_key,
                'q_values': {
                    'buy': float(q_values[0]),
                    'hold': float(q_values[1]),
                    'sell': float(q_values[2])
                },
                'best_action': action_names[best_action],
                'confidence': float(max(q_values))
            })
        else:
            return jsonify({'error': 'Q-table pickle not found'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    # Initialize database
    init_db()
    migrate_rl_training_jobs()
    # New: Setup and start the background scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=update_all_holdings_prices,
        trigger="interval",
        seconds=60,
        id='price_update_job'
    )
    scheduler.start()
    print("--- Live Data Scheduler started (updates every 60 seconds).")
    
    # Run the Flask app
    print("[INFO] Starting AI Trading Pro Backend Server...")
    print("[DB] Database: postgresql://ai_trading_pro:***@localhost:5432/ai_trading_db")
    print("[INFO] Server running on http://localhost:5000")
    print("\n--- API Endpoints:")
    print("   - GET  /api/health - Health check")
    print("   - GET  /api/users - Get all users")
    print("   - GET  /api/holdings/<user_id> - Get user holdings")
    print("   - POST /api/trades/<user_id> - Execute trade")
    print("   - GET  /api/watchlist/<user_id> - Get watchlist")
    print("   - POST /api/metrics/<user_id> - Save metrics")
    print("   - GET  /api/rl-agent/<user_id>/state - Get RL state")
    print("\n--- Frontend Pages:")
    print("   - GET  / - Dashboard (index.html)")
    print("   - GET  /trading - Trading (trading.html)")
    print("   - GET  /q_learning - Q-Learning (Q_learning.html)")
    print("   - GET  /news - News (news.html)")
    print("   - GET  /portfolio - Portfolio (portfolio.html)")
    print("   - GET  /analytics - Analytics (analytics.html)")
    print("\n--- Ready to accept requests!\n")

    app.run(debug=True, host='0.0.0.0', port=8000, use_reloader=False)
    