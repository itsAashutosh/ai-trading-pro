"""
Q-Learning Agent for Trading
"""

import numpy as np
import pickle
import os


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.
    """
    
    def __init__(self, action_size=3, learning_rate=0.1, discount_factor=0.95,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            action_size: Number of possible actions (3: HOLD, BUY, SELL)
            learning_rate: Alpha parameter for Q-learning
            discount_factor: Gamma parameter for future rewards
            exploration_rate: Initial epsilon for epsilon-greedy
            exploration_decay: Decay rate for epsilon
            min_exploration: Minimum epsilon value
        """
        self.action_size = action_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.min_epsilon = min_exploration
        
        # Q-table: dictionary mapping (state) -> [Q(s,a0), Q(s,a1), Q(s,a2)]
        self.q_table = {}
    
    def get_q_values(self, state):
        """Get Q-values for a state, initializing if necessary."""
        if state not in self.q_table:
            # Initialize Q-values to small random values
            self.q_table[state] = np.random.uniform(low=-0.01, high=0.01, size=self.action_size)
        return self.q_table[state]
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
            
        Returns:
            action: Integer (0=HOLD, 1=BUY, 2=SELL)
        """
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Exploitation: best action
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state (None if terminal)
            done: Whether episode is finished
        """
        # Get current Q-value
        q_values = self.get_q_values(state)
        current_q = q_values[action]
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_next_q = np.max(next_q_values)
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        q_values[action] = current_q + self.alpha * (target_q - current_q)
        self.q_table[state] = q_values
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save Q-table to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'q_table': self.q_table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'action_size': self.action_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ Q-table saved to {filepath}")
        print(f"   Total states learned: {len(self.q_table)}")
    
    def load(self, filepath):
        """Load Q-table from file."""
        if not os.path.exists(filepath):
            print(f"⚠️  Q-table file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data['q_table']
        self.alpha = save_data['alpha']
        self.gamma = save_data['gamma']
        self.epsilon = save_data['epsilon']
        self.epsilon_decay = save_data['epsilon_decay']
        self.min_epsilon = save_data['min_epsilon']
        self.action_size = save_data['action_size']
        
        print(f"✅ Q-table loaded from {filepath}")
        print(f"   Total states learned: {len(self.q_table)}")
        return True
    
    def get_policy(self):
        """
        Get the current policy (best action for each state).
        
        Returns:
            Dictionary mapping state -> best_action
        """
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = np.argmax(q_values)
        return policy
    
    def get_stats(self):
        """Get agent statistics."""
        return {
            'total_states': len(self.q_table),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
