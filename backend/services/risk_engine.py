from datetime import datetime
from backend.models.trading import Trade

class RiskEngine:
    MAX_TRADE_VALUE = 10000.0
    RESTRICTED_SYMBOLS = ['SCAM', 'FRAUD']
    
    @staticmethod
    def check_pre_trade_risk(user, trade_data):
        """
        Perform pre-trade risk checks.
        
        Args:
            user (User): The user object
            trade_data (dict): The trade details
            
        Returns:
            tuple: (passed: bool, reason: str)
        """
        symbol = trade_data.get('symbol', '').upper()
        quantity = int(trade_data.get('quantity', 0))
        price = float(trade_data.get('price', 0.0))
        total_value = quantity * price
        action = trade_data.get('action', '').upper()
        
        # 1. Restricted Symbols
        if symbol in RiskEngine.RESTRICTED_SYMBOLS:
            return False, f"Trading {symbol} is restricted."
            
        # 2. Max Trade Value
        if total_value > RiskEngine.MAX_TRADE_VALUE:
            return False, f"Trade value ${total_value} exceeds maximum limit of ${RiskEngine.MAX_TRADE_VALUE}."
            
        # 3. Short Selling Check (Simplified)
        if action == 'SELL':
            # This is covered by TradingEngine logic (insufficient shares), 
            # but a risk engine might enforce stricter rules like "No shorting at all".
            pass

        return True, "Risk checks passed"
