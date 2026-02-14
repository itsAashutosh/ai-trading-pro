"""
Trading Environment for Q-Learning Agent
Compatible with OpenAI Gym interface
"""

import numpy as np


class TradingEnv:
    """
    Trading environment for reinforcement learning.
    
    State: [price_change, position, balance_ratio]
    Actions: 0=HOLD, 1=BUY, 2=SELL
    """
    
    def __init__(self, data, initial_balance=10000, tx_cost=0.0):
        """
        Initialize trading environment.
        
        Args:
            data: Array of stock prices (Close prices)
            initial_balance: Starting cash balance
            tx_cost: Transaction cost as a fraction (e.g., 0.001 for 0.1%)
        """
        self.data = np.array(data)
        self.initial_balance = initial_balance
        self.tx_cost = tx_cost
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # Number of shares held
        self.total_trades = 0
        
        # For reward calculation
        self.last_portfolio_value = initial_balance
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.total_trades = 0
        self.last_portfolio_value = self.initial_balance
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation.
        
        Returns:
            Tuple of (price_change, position_flag, balance_ratio)
        """
        if self.current_step == 0:
            price_change = 0
        else:
            price_change = (self.data[self.current_step] - self.data[self.current_step - 1]) / self.data[self.current_step - 1]
        
        # Discretize position (0 = no position, 1 = has position)
        position_flag = 1 if self.position > 0 else 0
        
        # Current portfolio value
        current_price = self.data[self.current_step]
        portfolio_value = self.balance + self.position * current_price
        
        # Balance ratio (cash to total portfolio value)
        balance_ratio = self.balance / portfolio_value if portfolio_value > 0 else 0
        
        # Discretize for Q-table
        price_change_bucket = self._discretize_price_change(price_change)
        balance_ratio_bucket = self._discretize_balance_ratio(balance_ratio)
        
        return (price_change_bucket, position_flag, balance_ratio_bucket)
    
    def _discretize_price_change(self, price_change):
        """Discretize price change into buckets."""
        if price_change < -0.02:
            return 0  # Large decrease
        elif price_change < -0.005:
            return 1  # Small decrease
        elif price_change < 0.005:
            return 2  # Flat
        elif price_change < 0.02:
            return 3  # Small increase
        else:
            return 4  # Large increase
    
    def _discretize_balance_ratio(self, balance_ratio):
        """Discretize balance ratio into buckets."""
        if balance_ratio < 0.25:
            return 0  # Mostly invested
        elif balance_ratio < 0.5:
            return 1  # Half invested
        elif balance_ratio < 0.75:
            return 2  # Mostly cash
        else:
            return 3  # All cash
    
    def step(self, action):
        """
        Execute action and return next state.
        
        Args:
            action: 0=HOLD, 1=BUY, 2=SELL
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        current_price = self.data[self.current_step]
        
        # Execute action
        if action == 1:  # BUY
            # Buy as many shares as possible
            max_shares = int(self.balance / (current_price * (1 + self.tx_cost)))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.tx_cost)
                self.balance -= cost
                self.position += max_shares
                self.total_trades += 1
        
        elif action == 2:  # SELL
            # Sell all shares
            if self.position > 0:
                revenue = self.position * current_price * (1 - self.tx_cost)
                self.balance += revenue
                self.position = 0
                self.total_trades += 1
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Calculate reward
        if not done:
            next_price = self.data[self.current_step]
            portfolio_value = self.balance + self.position * next_price
        else:
            # Liquidate at end
            final_price = self.data[self.current_step]
            portfolio_value = self.balance + self.position * final_price
        
        # Reward is change in portfolio value
        reward = portfolio_value - self.last_portfolio_value
        self.last_portfolio_value = portfolio_value
        
        # Get next state
        next_state = self._get_state() if not done else None
        
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades
        }
        
        return next_state, reward, done, info
    
    def render(self):
        """Print current state (for debugging)."""
        current_price = self.data[self.current_step]
        portfolio_value = self.balance + self.position * current_price
        
        print(f"Step: {self.current_step}/{len(self.data)-1}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position} shares")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print("-" * 40)